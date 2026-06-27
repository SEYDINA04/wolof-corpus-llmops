# src/corpus_stats.py
"""
Statistiques du corpus wolof centralisé.

Calcule pour le dataset (parquet HF) :
  - nombre d'exemples
  - nombre de tokens (split whitespace) et de caractères
  - répartition par source
  - détection de langue (LID) avec GlotLID (cis-lmu/glotlid)
    -> distribution des langues, % wolof (wol_Latn), global + par source

Sorties :
  - affichage console
  - data/stats/corpus_stats.json         (rapport complet)
  - data/stats/corpus_stats_by_source.csv (résumé par source)

Usage :
  .venv/bin/python corpus_stats.py
  .venv/bin/python corpus_stats.py --parquet path/to.parquet --no-lid
  .venv/bin/python corpus_stats.py --sample 20000   # LID sur un échantillon
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DEFAULT_PARQUET = Path(__file__).parent / "wolof_centalized_corpus" / "data" / "train-00000-of-00001.parquet"
OUT_DIR = Path(__file__).parent / "data" / "stats"
WOLOF_LABEL = "wol_Latn"


# --------------------------------------------------------------------------- #
# GlotLID
# --------------------------------------------------------------------------- #
def load_glotlid():
    """Charge le modèle GlotLID (téléchargé depuis le Hub si besoin)."""
    import fasttext
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
    return fasttext.load_model(model_path)


def predict_lang(model, text: str):
    """Retourne (label, confiance) pour le top-1. Label ex: 'wol_Latn'."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ("__empty__", 0.0)
    text = str(text)
    # fastText ne supporte pas les retours à la ligne dans l'entrée
    clean = text.replace("\n", " ").replace("\r", " ").strip()
    if not clean:
        return ("__empty__", 0.0)
    # Appel direct au prédicteur C++ pour éviter le bug numpy 2.x
    # (model.predict utilise np.array(..., copy=False) -> ValueError).
    # self.f.predict attend une ligne terminée par '\n' et renvoie [(prob, label), ...]
    predictions = model.f.predict(clean + "\n", 1, 0.0, "strict")
    if not predictions:
        return ("__empty__", 0.0)
    prob, label = predictions[0]
    return (label.replace("__label__", ""), float(prob))


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def iter_sources(sources_value):
    """Normalise la colonne 'sources' (liste, str, None) en liste de str."""
    if sources_value is None:
        return ["__unknown__"]
    if isinstance(sources_value, str):
        return [sources_value]
    try:
        lst = list(sources_value)
    except TypeError:
        return [str(sources_value)]
    return lst if lst else ["__unknown__"]


def compute_stats(df: pd.DataFrame, model=None, sample: int | None = None):
    n = len(df)

    # --- Comptages de base (sur tout le corpus) ---
    total_tokens = 0
    total_chars = 0
    empty_count = 0

    # par source : exemples, tokens, chars
    src_examples = Counter()
    src_tokens = Counter()
    src_chars = Counter()

    for text, sources in tqdm(
        zip(df["text"], df["sources"]), total=n, desc="Comptage tokens/chars"
    ):
        text = "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)
        ntok = len(text.split())
        nchar = len(text)
        total_tokens += ntok
        total_chars += nchar
        if not text.strip():
            empty_count += 1
        for s in iter_sources(sources):
            src_examples[s] += 1
            src_tokens[s] += ntok
            src_chars[s] += nchar

    report = {
        "parquet_rows": n,
        "total_examples": n,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "empty_examples": empty_count,
        "avg_tokens_per_example": round(total_tokens / n, 2) if n else 0,
        "avg_chars_per_example": round(total_chars / n, 2) if n else 0,
        "by_source": {},
        "lid": None,
    }

    for s in sorted(src_examples, key=lambda k: -src_examples[k]):
        ex = src_examples[s]
        report["by_source"][s] = {
            "examples": ex,
            "tokens": src_tokens[s],
            "chars": src_chars[s],
            "avg_tokens": round(src_tokens[s] / ex, 2) if ex else 0,
        }

    # --- LID GlotLID ---
    if model is not None:
        lid_df = df
        sampled = False
        if sample and sample < n:
            lid_df = df.sample(n=sample, random_state=42)
            sampled = True

        lang_global = Counter()
        lang_by_source = defaultdict(Counter)
        wolof_conf_sum = 0.0
        wolof_n = 0

        for text, sources in tqdm(
            zip(lid_df["text"], lid_df["sources"]),
            total=len(lid_df),
            desc="LID GlotLID",
        ):
            label, conf = predict_lang(model, text or "")
            lang_global[label] += 1
            for s in iter_sources(sources):
                lang_by_source[s][label] += 1
            if label == WOLOF_LABEL:
                wolof_conf_sum += conf
                wolof_n += 1

        total_lid = sum(lang_global.values())
        report["lid"] = {
            "model": "cis-lmu/glotlid",
            "wolof_label": WOLOF_LABEL,
            "sampled": sampled,
            "n_evaluated": total_lid,
            "wolof_count": lang_global.get(WOLOF_LABEL, 0),
            "wolof_pct": round(100 * lang_global.get(WOLOF_LABEL, 0) / total_lid, 2)
            if total_lid
            else 0,
            "wolof_avg_confidence": round(wolof_conf_sum / wolof_n, 4) if wolof_n else 0,
            "top_languages": [
                {"lang": lang, "count": c, "pct": round(100 * c / total_lid, 2)}
                for lang, c in lang_global.most_common(20)
            ],
            "wolof_pct_by_source": {},
        }
        for s, counts in lang_by_source.items():
            tot = sum(counts.values())
            report["lid"]["wolof_pct_by_source"][s] = {
                "evaluated": tot,
                "wolof": counts.get(WOLOF_LABEL, 0),
                "wolof_pct": round(100 * counts.get(WOLOF_LABEL, 0) / tot, 2)
                if tot
                else 0,
                "top_lang": counts.most_common(1)[0][0] if counts else None,
            }

    return report


# --------------------------------------------------------------------------- #
# Affichage + sauvegarde
# --------------------------------------------------------------------------- #
def print_report(report: dict):
    print("\n" + "=" * 60)
    print("STATISTIQUES DU CORPUS WOLOF CENTRALISÉ")
    print("=" * 60)
    print(f"Exemples           : {report['total_examples']:,}")
    print(f"Tokens (total)     : {report['total_tokens']:,}")
    print(f"Caractères (total) : {report['total_chars']:,}")
    print(f"Tokens / exemple   : {report['avg_tokens_per_example']}")
    print(f"Exemples vides     : {report['empty_examples']:,}")

    print("\n--- Par source ---")
    print(f"{'source':45} {'exemples':>10} {'tokens':>14} {'tok/ex':>8}")
    for s, d in report["by_source"].items():
        print(f"{s[:45]:45} {d['examples']:>10,} {d['tokens']:>14,} {d['avg_tokens']:>8}")

    lid = report.get("lid")
    if lid:
        print("\n--- LID (GlotLID) ---")
        if lid["sampled"]:
            print(f"(échantillon de {lid['n_evaluated']:,} exemples)")
        print(
            f"Wolof (wol_Latn)   : {lid['wolof_count']:,} / {lid['n_evaluated']:,} "
            f"= {lid['wolof_pct']}%  (conf. moy. {lid['wolof_avg_confidence']})"
        )
        print("\nTop langues détectées :")
        for row in lid["top_languages"]:
            print(f"  {row['lang']:14} {row['count']:>10,}  {row['pct']:>6}%")

        print("\n% wolof par source :")
        print(f"{'source':45} {'eval':>8} {'wolof':>8} {'%wol':>7} {'top':>10}")
        for s, d in sorted(
            lid["wolof_pct_by_source"].items(), key=lambda kv: -kv[1]["evaluated"]
        ):
            print(
                f"{s[:45]:45} {d['evaluated']:>8,} {d['wolof']:>8,} "
                f"{d['wolof_pct']:>6}% {str(d['top_lang']):>10}"
            )
    print("=" * 60 + "\n")


def save_report(report: dict, out_dir: Path = OUT_DIR, name: str = "corpus_stats"):
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / f"{name}_by_source.csv"
    lid = report.get("lid") or {}
    lid_src = lid.get("wolof_pct_by_source", {})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "examples", "tokens", "chars", "avg_tokens", "wolof_pct", "top_lang"])
        for s, d in report["by_source"].items():
            ls = lid_src.get(s, {})
            w.writerow([
                s, d["examples"], d["tokens"], d["chars"], d["avg_tokens"],
                ls.get("wolof_pct", ""), ls.get("top_lang", ""),
            ])

    print(f"Rapport JSON : {json_path}")
    print(f"Résumé CSV   : {csv_path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Stats du corpus wolof centralisé")
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--no-lid", action="store_true", help="Désactiver la détection de langue")
    ap.add_argument("--sample", type=int, default=None, help="LID sur un échantillon de N exemples")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Dossier de sortie des rapports")
    ap.add_argument("--out-name", type=str, default="corpus_stats", help="Préfixe des fichiers de sortie")
    args = ap.parse_args()

    print(f"Chargement : {args.parquet}")
    df = pd.read_parquet(args.parquet)
    print(f"  -> {len(df):,} lignes, colonnes = {list(df.columns)}")

    model = None
    if not args.no_lid:
        print("Chargement du modèle GlotLID...")
        model = load_glotlid()

    report = compute_stats(df, model=model, sample=args.sample)
    print_report(report)
    save_report(report, out_dir=args.out_dir, name=args.out_name)


if __name__ == "__main__":
    main()
