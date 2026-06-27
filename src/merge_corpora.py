# src/merge_corpora.py
"""
Fusion dédupliquée du corpus central et du corpus ingéré vers un NOUVEAU
corpus unifié, au même format parquet (colonnes `text` str, `sources` liste).

  - lit les 2 parquet
  - déduplication exacte (clé = text.lower().strip())
    -> pour un texte présent des deux côtés, les listes `sources` sont fusionnées
  - écrit wolof_unified_corpus/data/train-00000-of-00001.parquet

Les corpus d'origine ne sont pas modifiés.

Usage :
  .venv/bin/python merge_corpora.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
CENTRAL = HERE / "wolof_centalized_corpus" / "data" / "train-00000-of-00001.parquet"
INGEST = HERE / "wolof_ingested_corpus" / "data" / "train-00000-of-00001.parquet"
OUT_DIR = HERE / "wolof_unified_corpus" / "data"
OUT_PARQUET = OUT_DIR / "train-00000-of-00001.parquet"


def norm_sources(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(s) for s in value]
    except TypeError:
        return [str(value)]


def add_df(df, merged, stats, name):
    n = 0
    for text, sources in zip(df["text"], df["sources"]):
        if text is None or (isinstance(text, float) and pd.isna(text)):
            continue
        text = str(text)
        key = text.lower().strip()
        if not key:
            continue
        n += 1
        srcs = norm_sources(sources)
        if key in merged:
            stats["dup"] += 1
            merged[key]["sources"].update(srcs)
        else:
            merged[key] = {"text": text, "sources": set(srcs)}
    print(f"  {name:20} : {n:,} lignes lues")


def main():
    print(f"Central : {CENTRAL}")
    print(f"Ingéré  : {INGEST}\n")
    dfc = pd.read_parquet(CENTRAL)
    dfi = pd.read_parquet(INGEST)

    merged: dict[str, dict] = {}
    stats = {"dup": 0}

    add_df(dfc, merged, stats, "central")
    n_after_central = len(merged)
    add_df(dfi, merged, stats, "ingéré")

    rows = [{"text": v["text"], "sources": sorted(v["sources"])} for v in merged.values()]
    df = pd.DataFrame(rows, columns=["text", "sources"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    print("\n" + "=" * 60)
    print("FUSION TERMINÉE")
    print("=" * 60)
    print(f"Central (lignes)            : {len(dfc):,}")
    print(f"Ingéré  (lignes)            : {len(dfi):,}")
    print(f"Doublons fusionnés          : {stats['dup']:,}")
    print(f"Exemples uniques (unifié)   : {len(df):,}")
    multi = df["sources"].map(len)
    print(f"Exemples multi-sources      : {(multi > 1).sum():,}")
    print(f"💾 {OUT_PARQUET}")


if __name__ == "__main__":
    main()
