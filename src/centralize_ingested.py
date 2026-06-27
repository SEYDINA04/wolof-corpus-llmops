# src/centralize_ingested.py
"""
Centralise les .jsonl produits par ingest_hf_datasets.py (src/data/ingested/)
dans un corpus *séparé* respectant le même format que le corpus central :
  - parquet avec colonnes `text` (str) et `sources` (liste de str)

Étapes :
  - lit tous les data/ingested/*.jsonl  ({text, sources:[repo]})
  - déduplication exacte inter-datasets (clé = text.lower().strip())
    -> un même texte présent dans plusieurs datasets fusionne ses sources
  - écrit wolof_ingested_corpus/data/train-00000-of-00001.parquet

Usage :
  .venv/bin/python centralize_ingested.py
"""

import json
from pathlib import Path

import pandas as pd

ING_DIR = Path(__file__).parent / "data" / "ingested"
OUT_DIR = Path(__file__).parent / "wolof_ingested_corpus" / "data"
OUT_PARQUET = OUT_DIR / "train-00000-of-00001.parquet"


def main():
    files = sorted(ING_DIR.glob("*.jsonl"))
    print(f"{len(files)} fichiers .jsonl trouvés dans {ING_DIR}")

    merged: dict[str, dict] = {}  # key -> {"text": ..., "sources": set()}
    total_lines = 0
    dup_cross = 0

    for fp in files:
        n = 0
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj["text"]
                srcs = obj.get("sources") or []
                key = text.lower().strip()
                n += 1
                total_lines += 1
                if key in merged:
                    dup_cross += 1
                    merged[key]["sources"].update(srcs)
                else:
                    merged[key] = {"text": text, "sources": set(srcs)}
        print(f"  {fp.name:55} {n:>8,} lignes")

    rows = [
        {"text": v["text"], "sources": sorted(v["sources"])}
        for v in merged.values()
    ]
    df = pd.DataFrame(rows, columns=["text", "sources"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    print("\n" + "=" * 60)
    print("CENTRALISATION TERMINÉE")
    print("=" * 60)
    print(f"Lignes lues (brut)        : {total_lines:,}")
    print(f"Doublons inter-datasets   : {dup_cross:,}")
    print(f"Exemples uniques (corpus) : {len(df):,}")
    print(f"Colonnes                  : {list(df.columns)}")
    print(f"💾 {OUT_PARQUET}")


if __name__ == "__main__":
    main()
