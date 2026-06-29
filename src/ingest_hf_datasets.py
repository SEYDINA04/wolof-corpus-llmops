# src/ingest_hf_datasets.py
"""
Ingestion de datasets HuggingFace wolof vers le format du corpus centralisé.

Pour chaque dataset configuré :
  - charge le(s) split(s)
  - extrait la/les colonne(s) wolof -> {text, sources: [repo_id]}
  - filtre : longueur min (tokens) + détection de langue GlotLID (wol_Latn >= seuil)
  - déduplication interne (exacte, lowercase+trim)
  - écrit src/data/ingested/<repo_id_safe>.jsonl

Usage :
  .venv/bin/python ingest_hf_datasets.py                 # tous les datasets
  .venv/bin/python ingest_hf_datasets.py --only soynade-research/FineWeb2-HQ-50k-Wolof
  .venv/bin/python ingest_hf_datasets.py --skip-existing  # reprise (ne refait pas les jsonl déjà produits)
  .venv/bin/python ingest_hf_datasets.py --no-lid         # sans filtre de langue
  .venv/bin/python ingest_hf_datasets.py --lid-threshold 0.5
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

OUT_DIR = Path(__file__).parent / "data" / "ingested"
WOLOF_LABEL = "wol_Latn"
MIN_TOKENS = 3
DEFAULT_LID_THRESHOLD = 0.5

# --------------------------------------------------------------------------- #
# Config : (repo_id, [colonnes wolof], split)
# Seules les colonnes listées (texte wolof) sont extraites ; chaque colonne
# non vide d'une ligne produit un exemple distinct.
# --------------------------------------------------------------------------- #
DATASETS = [
    # --- texte natif / web ---
    {"repo": "soynade-research/FineWeb2-HQ-50k-Wolof", "cols": ["wolof"], "split": "train"},
    {"repo": "soynade-research/Wolof-Non-Standard-Orthography", "cols": ["wo", "non_standardized"], "split": "train"},
    # --- traductions (côté wolof) ---
    {"repo": "ZigZeug/Baatukaay-wolof-translated-dataset", "cols": ["wo"], "split": "train"},
    {"repo": "bilalfaye/english-wolof-french-dataset", "cols": ["wo"], "split": "train"},
    {"repo": "bilalfaye/wolof-english-french", "cols": ["wo"], "split": "train"},
    {"repo": "Bassoumm/wolof-french-dictionary", "cols": ["wolof"], "split": "train"},
    {"repo": "skonteye/French-Wolof-Dataset-With-Sources", "cols": ["wolof"], "split": "train"},
    {"repo": "galsenai/english-wolof-smol-translation", "cols": ["wo"], "split": "train"},
    {"repo": "MaroneAI/French-Wolof_Translation-Dataset", "cols": ["target"], "split": "train"},
    {"repo": "MaroneAI/Wolof-to-French_Translation-Dataset", "cols": ["Input"], "split": "train"},
    {"repo": "mbaye930/wolof-arabic-parallel-corpus", "cols": ["wolof"], "split": "train"},
    # --- autres textes ---
    {"repo": "soynade-research/Wolof-Agri-Captions", "cols": ["wolof"], "split": "train"},
    {"repo": "vonewman/fleurs-wolof-dataset", "cols": ["transcription"], "split": "train"},
    {"repo": "michsethowusu/wolof-sentiments-corpus", "cols": ["Wolof"], "split": "train"},
    {"repo": "mbaye930/WolofEntityLinking", "cols": ["text"], "split": "train", "lid": False, "min_tokens": 1},
    # --- instructions (LLM) ---
    {"repo": "m-a-d-i/wori-wolof-instructions", "cols": ["text_wo", "instruction_wo"], "split": "train"},
    {"repo": "ngia/alpaca-data-in-wolof", "cols": ["instruction", "output"], "split": "train"},
    # --- monolingue web / encyclopédique (gros volume, multi-configs) ---
    {"repo": "HPLT/HPLT2.0_cleaned", "cols": ["text"], "config": "wol_Latn", "split": "train"},
    {"repo": "cis-lmu/Glot500", "cols": ["text"], "config": "wol_Latn", "split": "train"},
    {"repo": "HuggingFaceFW/fineweb-2", "cols": ["text"], "config": "wol_Latn", "split": "train"},
    {"repo": "cis-lmu/GlotCC-V1", "cols": ["content"], "config": "wol-Latn", "split": "train"},
    {"repo": "aiana94/polynews", "cols": ["text"], "config": "wol_Latn", "split": "train"},
    {"repo": "Davlan/sib200", "cols": ["text"], "config": "wol_Latn", "split": "train"},
    # --- encyclopédique / religieux / parallèle (lot 250 MB) ---
    {"repo": "wikimedia/wikipedia", "cols": ["text"], "config": "20231101.wo", "split": "train"},
    {"repo": "alexandrainst/multi-wiki-qa", "cols": ["context"], "config": "wo", "split": "train"},
    {"repo": "Lahad/fr_wolof_quran_corpus", "cols": ["texte_wolof"], "split": "train"},
    {"repo": "dofbi/jolof", "cols": ["wolof"], "split": "train"},
    {"repo": "AfriNLP/AfriNLLB-train", "cols": ["source", "target"], "split": "train"},
    # --- gros volumes (objectif 1M) ---
    {"repo": "michsethowusu/wolof-emotions-corpus", "cols": ["Wolof"], "split": "train"},
    {"repo": "geekdiop/A-Wolof-Arabic-Parallel-Corpus", "cols": ["text"], "split": "train"},
    {"repo": "michsethowusu/Code-170k-wolof", "cols": ["conversations"], "split": "train"},  # ShareGPT: tours wolof
]


# --------------------------------------------------------------------------- #
# GlotLID
# --------------------------------------------------------------------------- #
def load_glotlid():
    import fasttext
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
    return fasttext.load_model(model_path)


def predict_lang(model, text: str):
    """Retourne (label, confiance). Appel direct au C++ (bug numpy 2.x avec .predict)."""
    clean = text.replace("\n", " ").replace("\r", " ").strip()
    if not clean:
        return ("__empty__", 0.0)
    predictions = model.f.predict(clean + "\n", 1, 0.0, "strict")
    if not predictions:
        return ("__empty__", 0.0)
    prob, label = predictions[0]
    return (label.replace("__label__", ""), float(prob))


# --------------------------------------------------------------------------- #
def safe_name(repo: str) -> str:
    return repo.replace("/", "__")


def norm_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        try:
            import math
            if math.isnan(value):
                return ""
        except Exception:
            pass
    return str(value).strip()


def extract_texts(value) -> list[str]:
    """Extrait un ou plusieurs textes d'une cellule.

    - chaîne simple -> [texte]
    - conversations (liste de {"from", "value"}) -> [value, value, ...]
      (format ShareGPT, ex. michsethowusu/Code-170k-wolof)
    """
    if isinstance(value, list):
        out = []
        for turn in value:
            if isinstance(turn, dict):
                out.append(norm_text(turn.get("value", "")))
            else:
                out.append(norm_text(turn))
        return [t for t in out if t]
    t = norm_text(value)
    return [t] if t else []


def ingest_one(cfg: dict, model, lid_threshold: float, min_tokens: int):
    from datasets import load_dataset

    repo = cfg["repo"]
    cols = cfg["cols"]
    split = cfg.get("split", "train")

    # Overrides par dataset (sources de confiance, annotées manuellement) :
    #   "lid": false        -> ne pas appliquer le filtre de langue GlotLID
    #   "min_tokens": N      -> seuil de longueur spécifique
    eff_model = None if cfg.get("lid") is False else model
    eff_min_tokens = int(cfg.get("min_tokens", min_tokens))
    trust = cfg.get("lid") is False or "min_tokens" in cfg

    badge = "  [source de confiance: filtres relâchés]" if trust else ""
    config = cfg.get("config")
    cfg_txt = f"  config={config}" if config else ""
    print(f"\n{'='*60}\n📥 {repo}{cfg_txt}  (colonnes: {cols}){badge}\n{'='*60}")

    # On charge en gardant uniquement les colonnes texte voulues si possible.
    # `config` (ex: 'wol_Latn') sélectionne le sous-ensemble d'un dataset multilingue.
    ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
    available = set(ds.column_names)
    use_cols = [c for c in cols if c in available]
    if not use_cols:
        print(f"  ⚠️  Aucune des colonnes {cols} trouvée. Colonnes dispo: {sorted(available)}")
        return None

    # Drop des colonnes lourdes (audio/image) pour éviter de matérialiser des Go
    drop = [c for c in available if c not in use_cols]
    if drop:
        ds = ds.remove_columns([c for c in drop if c in ds.column_names])

    stats = {
        "repo": repo,
        "cols": use_cols,
        "rows_input": ds.num_rows,
        "candidates": 0,
        "rejected_short": 0,
        "rejected_lang": 0,
        "rejected_dup": 0,
        "kept": 0,
        "lang_counts": {},
    }

    seen = set()
    kept_items = []

    for row in tqdm(ds, total=ds.num_rows, desc=f"  {safe_name(repo)}"):
        for c in use_cols:
            for text in extract_texts(row.get(c)):
                stats["candidates"] += 1

                if len(text.split()) < eff_min_tokens:
                    stats["rejected_short"] += 1
                    continue

                if eff_model is not None:
                    label, conf = predict_lang(eff_model, text)
                    stats["lang_counts"][label] = stats["lang_counts"].get(label, 0) + 1
                    if label != WOLOF_LABEL or conf < lid_threshold:
                        stats["rejected_lang"] += 1
                        continue

                key = text.lower()
                if key in seen:
                    stats["rejected_dup"] += 1
                    continue
                seen.add(key)

                kept_items.append({"text": text, "sources": [repo]})

    stats["kept"] = len(kept_items)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{safe_name(repo)}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for item in kept_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # top langues rejetées (info)
    top_lang = sorted(stats["lang_counts"].items(), key=lambda kv: -kv[1])[:5]
    print(
        f"  → input={stats['rows_input']:,}  candidats={stats['candidates']:,}  "
        f"gardés={stats['kept']:,}\n"
        f"     rejets: courts={stats['rejected_short']:,} "
        f"langue={stats['rejected_lang']:,} dups={stats['rejected_dup']:,}"
    )
    if top_lang:
        print("     langues détectées (top5): " + ", ".join(f"{l}={n}" for l, n in top_lang))
    print(f"     💾 {out_path}")

    return stats


def main():
    ap = argparse.ArgumentParser(description="Ingestion datasets HF wolof")
    ap.add_argument("--only", action="append", help="N'ingérer que ce(s) repo_id (répétable)")
    ap.add_argument("--skip-existing", action="store_true", help="Ignorer les jsonl déjà produits")
    ap.add_argument("--no-lid", action="store_true", help="Désactiver le filtre de langue GlotLID")
    ap.add_argument("--lid-threshold", type=float, default=DEFAULT_LID_THRESHOLD)
    ap.add_argument("--min-tokens", type=int, default=MIN_TOKENS)
    args = ap.parse_args()

    targets = DATASETS
    if args.only:
        wanted = set(args.only)
        targets = [d for d in DATASETS if d["repo"] in wanted]
        if not targets:
            print(f"Aucun dataset configuré ne correspond à {args.only}")
            return

    model = None
    if not args.no_lid:
        print("Chargement du modèle GlotLID...")
        model = load_glotlid()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = []
    for cfg in targets:
        out_path = OUT_DIR / f"{safe_name(cfg['repo'])}.jsonl"
        if args.skip_existing and out_path.exists():
            print(f"⏭️  {cfg['repo']} déjà ingéré ({out_path.name}), skip.")
            continue
        try:
            st = ingest_one(cfg, model, args.lid_threshold, args.min_tokens)
            if st:
                all_stats.append(st)
        except Exception as e:
            print(f"  ❌ ERREUR sur {cfg['repo']}: {type(e).__name__}: {e}")

    # Récap global
    print("\n" + "=" * 60)
    print("RÉCAP INGESTION")
    print("=" * 60)
    print(f"{'dataset':50} {'gardés':>10}")
    total = 0
    for st in all_stats:
        total += st["kept"]
        print(f"{st['repo'][:50]:50} {st['kept']:>10,}")
    print("-" * 62)
    print(f"{'TOTAL gardés':50} {total:>10,}")

    # Sauvegarde du récap
    report_path = OUT_DIR / "_ingestion_report.json"
    report_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRapport: {report_path}")


if __name__ == "__main__":
    main()
