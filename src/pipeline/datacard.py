# src/pipeline/datacard.py
"""
Génère automatiquement la *data card* (README.md) du dataset HF à partir :
  - du parquet final (métadonnées : nb exemples, tailles)
  - du rapport de stats (corpus_stats du corpus unifié)

La carte contient le front-matter YAML attendu par HuggingFace (dataset_info)
+ une description lisible (sources, volumétrie, qualité langue).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def _raw_text_bytes(df: pd.DataFrame) -> int:
    return int(df["text"].fillna("").astype(str).map(lambda s: len(s.encode("utf-8"))).sum())


def build_readme(parquet_path: str | Path, stats: dict | None) -> str:
    df = pd.read_parquet(parquet_path)
    n = len(df)
    download_size = os.path.getsize(parquet_path)

    # taille Arrow décompressée (dataset_size)
    try:
        from datasets import Dataset

        num_bytes = Dataset.from_pandas(df, preserve_index=False).data.nbytes
    except Exception:
        num_bytes = _raw_text_bytes(df) * 2  # approximation de repli

    total_tokens = stats.get("total_tokens") if stats else None
    wolof_pct = stats.get("lid", {}).get("wolof_pct") if stats else None
    by_source = stats.get("by_source", {}) if stats else {}

    front = f"""---
dataset_info:
  features:
  - name: text
    dtype: string
  - name: sources
    list: string
  splits:
  - name: train
    num_bytes: {num_bytes}
    num_examples: {n}
  download_size: {download_size}
  dataset_size: {num_bytes}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
language:
- wo
license: cc-by-4.0
task_categories:
- text-generation
- translation
size_categories:
- 100K<n<1M
---"""

    lines = [
        front,
        "",
        "# Wolof Centralized Corpus",
        "",
        "Corpus wolof centralisé et dédupliqué, agrégeant de multiples sources "
        "publiques (texte web, traductions, ASR, instructions LLM).",
        "",
        "## Aperçu",
        "",
        f"- **Exemples** : {n:,}",
    ]
    if total_tokens:
        lines.append(f"- **Tokens** (approx. whitespace) : {total_tokens:,}")
    lines.append(f"- **Texte brut** : {_raw_text_bytes(df) / 1_000_000:.1f} MB (UTF-8)")
    if wolof_pct is not None:
        lines.append(f"- **Part de wolof** (GlotLID, échantillon) : {wolof_pct}%")
    lines += [
        "",
        "## Schéma",
        "",
        "| colonne | type | description |",
        "|---|---|---|",
        "| `text` | string | texte wolof |",
        "| `sources` | list[string] | identifiants des datasets d'origine |",
        "",
    ]

    if by_source:
        lines += [
            "## Répartition par source",
            "",
            "| source | exemples | tokens |",
            "|---|---:|---:|",
        ]
        for s, d in sorted(by_source.items(), key=lambda kv: -kv[1]["examples"]):
            lines.append(f"| `{s}` | {d['examples']:,} | {d.get('tokens', 0):,} |")
        lines.append("")

    lines += [
        "## Construction",
        "",
        "Généré automatiquement par le pipeline LLMOps (`pipeline/run.py`) :",
        "ingestion HF → centralisation → fusion dédupliquée → stats → quality gates → publication.",
        "",
        "_Cette carte est régénérée à chaque publication._",
        "",
    ]
    return "\n".join(lines)


def write_readme(parquet_path: str | Path, stats: dict | None, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.write_text(build_readme(parquet_path, stats), encoding="utf-8")
    return out
