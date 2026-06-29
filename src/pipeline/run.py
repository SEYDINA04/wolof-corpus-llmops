# src/pipeline/run.py
"""
Orchestrateur du pipeline LLMOps du corpus wolof.

Stages (idempotents, exécutables séparément ou en chaîne) :

  ingest      [1] ingère les 17 datasets HF -> data/ingested/*.jsonl   (resumable)
  centralize  [2] fusionne les jsonl -> corpus ingéré (parquet, dédup interne)
  merge       [3] fusionne corpus HF + ingéré -> corpus unifié (dédup globale)
  stats       [4] calcule les statistiques (volumétrie + LID GlotLID)
  datacard    [5] régénère le README (data card) du dataset
  validate    [6] exécute les QUALITY GATES (bloquant)
  publish     [7] pousse parquet + README vers HF (refusé si gates KO)

Exemples :
  python -m pipeline.run all                 # tout sauf publish
  python -m pipeline.run validate            # uniquement les gates
  python -m pipeline.run publish             # publie (après validate)
  python -m pipeline.run all --publish       # bout en bout

Le pipeline lit toute sa config dans pipeline.yaml + secrets via .env.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from pipeline import datacard
from pipeline import quality_gates as qg
from pipeline.config import SRC_DIR, load_config

PYTHON = sys.executable  # .venv/bin/python courant


def _run(cmd: list[str], desc: str) -> None:
    print(f"\n{'─' * 70}\n▶ {desc}\n  $ {' '.join(cmd)}\n{'─' * 70}")
    res = subprocess.run(cmd, cwd=SRC_DIR)
    if res.returncode != 0:
        sys.exit(f"[ÉCHEC] stage '{desc}' code={res.returncode}")


# --------------------------------------------------------------------------- #
def stage_ingest(cfg) -> None:
    _run(
        [
            PYTHON,
            "ingest_hf_datasets.py",
            "--skip-existing",
            "--lid-threshold",
            str(cfg.ingest["lid_threshold"]),
            "--min-tokens",
            str(cfg.ingest["min_tokens"]),
        ],
        "INGEST — datasets HF -> data/ingested/",
    )


def stage_centralize(cfg) -> None:
    _run([PYTHON, "centralize_ingested.py"], "CENTRALIZE — jsonl -> corpus ingéré")


def stage_merge(cfg) -> None:
    # Source de vérité = le parquet ACTUELLEMENT publié sur HF.
    # On le télécharge dans le chemin `central_clone` pour que la fusion
    # parte toujours de l'état réel du dataset (cohérent en local ET en CI).
    print("Récupération du corpus central depuis HF (base de fusion)...")
    hf_df = qg.fetch_hf_dataframe(cfg.hf_repo, cfg.hf_filename, cfg.hf_repo_type, cfg.hf_token)
    if hf_df is None:
        sys.exit("[ERREUR] impossible de récupérer le corpus HF pour la fusion.")
    central_path = cfg.p("central_clone")
    central_path.parent.mkdir(parents=True, exist_ok=True)
    hf_df.to_parquet(central_path, index=False)
    print(f"  central téléchargé : {len(hf_df):,} lignes -> {central_path}")
    _run([PYTHON, "merge_corpora.py"], "MERGE — corpus HF + ingéré -> unifié")


def stage_stats(cfg) -> None:
    _run(
        [
            PYTHON,
            "corpus_stats.py",
            "--parquet",
            str(cfg.p("unified_corpus")),
            "--sample",
            str(cfg.stats["lid_sample"]),
            "--out-dir",
            str(cfg.p("stats_dir")),
            "--out-name",
            "unified_corpus_stats",
        ],
        "STATS — volumétrie + LID GlotLID",
    )


def stage_datacard(cfg) -> None:
    stats = qg.load_stats(cfg.p("stats_dir") / "unified_corpus_stats.json")
    out = datacard.write_readme(cfg.p("unified_corpus"), stats, cfg.p("datacard"))
    print(f"✅ data card régénérée -> {out}")


def stage_validate(cfg) -> qg.GateReport:
    stats = qg.load_stats(cfg.p("stats_dir") / "unified_corpus_stats.json")
    hf_path = None
    if cfg.gates.get("require_no_hf_loss"):
        print("Téléchargement du parquet HF actuel (contrôle anti-perte, streamé)...")
        hf_path = qg.fetch_hf_parquet_path(cfg.hf_repo, cfg.hf_filename, cfg.hf_repo_type, cfg.hf_token)
    report = qg.run_gates(cfg.p("unified_corpus"), cfg.gates, stats=stats, hf_parquet_path=hf_path)
    print("\n" + report.render())
    # trace JSON
    out = cfg.p("stats_dir") / "quality_gates_report.json"
    import json

    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRapport gates -> {out}")
    return report


def stage_publish(cfg) -> None:
    # garde-fou : on ne publie que si les gates passent
    report = stage_validate(cfg)
    if not report.all_passed:
        sys.exit("[REFUS] Quality gates non passés — publication annulée.")

    if not cfg.hf_token:
        sys.exit("[ERREUR] HF_TOKEN absent (configurer .env ou secret CI).")

    from huggingface_hub import HfApi

    api = HfApi(token=cfg.hf_token)
    who = api.whoami()
    print(f"Authentifié : {who.get('name')}  orgs={[o['name'] for o in who.get('orgs', [])]}")

    import pyarrow.parquet as pq

    n_rows = pq.ParquetFile(cfg.p("unified_corpus")).metadata.num_rows
    msg = f"Pipeline publish — {n_rows:,} exemples"
    print("Upload README.md...")
    api.upload_file(
        path_or_fileobj=str(cfg.p("datacard")),
        path_in_repo="README.md",
        repo_id=cfg.hf_repo,
        repo_type=cfg.hf_repo_type,
        commit_message=msg,
    )
    print("Upload parquet...")
    # on pousse le corpus unifié comme nouveau fichier central
    api.upload_file(
        path_or_fileobj=str(cfg.p("unified_corpus")),
        path_in_repo=cfg.hf_filename,
        repo_id=cfg.hf_repo,
        repo_type=cfg.hf_repo_type,
        commit_message=msg,
    )
    print(f"\n✅ PUBLIÉ -> https://huggingface.co/datasets/{cfg.hf_repo}")


STAGES = {
    "ingest": stage_ingest,
    "centralize": stage_centralize,
    "merge": stage_merge,
    "stats": stage_stats,
    "datacard": stage_datacard,
    "validate": stage_validate,
    "publish": stage_publish,
}
# ordre "all" (publish exclu par défaut)
ALL_ORDER = ["ingest", "centralize", "merge", "stats", "datacard", "validate"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline LLMOps corpus wolof")
    ap.add_argument("stage", choices=list(STAGES) + ["all"], help="étape à exécuter")
    ap.add_argument("--config", default=None, help="chemin pipeline.yaml")
    ap.add_argument("--publish", action="store_true", help="avec 'all' : publie à la fin")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    print(f"Config : {cfg.path}\nRepo HF: {cfg.hf_repo}")

    if args.stage == "all":
        for name in ALL_ORDER:
            STAGES[name](cfg)
        if args.publish:
            STAGES["publish"](cfg)
        print("\n✅ Pipeline terminé.")
    else:
        result = STAGES[args.stage](cfg)
        if isinstance(result, qg.GateReport) and not result.all_passed:
            sys.exit(1)


if __name__ == "__main__":
    main()
