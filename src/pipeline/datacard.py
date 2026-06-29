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

import pyarrow.compute as pc
import pyarrow.parquet as pq


def _scan_parquet(parquet_path: str | Path) -> tuple[int, int, int]:
    """Parcourt le parquet par batches (mémoire constante) et retourne :
    (nb_lignes, octets_arrow_décompressés, octets_texte_brut_utf8).

    On ne charge JAMAIS tout le corpus en mémoire d'un coup : le pipeline
    tournait OOM (machine figée) en matérialisant le parquet en pandas PUIS
    en le dupliquant via ``Dataset.from_pandas``. Ici un seul batch vit à la fois.
    """
    pf = pq.ParquetFile(parquet_path)
    n = pf.metadata.num_rows
    arrow_bytes = 0
    raw_text_bytes = 0
    text_idx = pf.schema_arrow.get_field_index("text")
    for batch in pf.iter_batches(batch_size=50_000):
        arrow_bytes += batch.nbytes
        text_col = batch.column(text_idx)
        s = pc.sum(pc.binary_length(text_col)).as_py()
        raw_text_bytes += int(s) if s is not None else 0
    return n, arrow_bytes, raw_text_bytes


def build_readme(parquet_path: str | Path, stats: dict | None) -> str:
    download_size = os.path.getsize(parquet_path)
    n, num_bytes, raw_text_bytes = _scan_parquet(parquet_path)

    total_tokens = stats.get("total_tokens") if stats else None
    wolof_pct = stats.get("lid", {}).get("wolof_pct") if stats else None
    by_source = stats.get("by_source", {}) if stats else {}

    if n < 1_000:
        size_cat = "n<1K"
    elif n < 10_000:
        size_cat = "1K<n<10K"
    elif n < 100_000:
        size_cat = "10K<n<100K"
    elif n < 1_000_000:
        size_cat = "100K<n<1M"
    elif n < 10_000_000:
        size_cat = "1M<n<10M"
    else:
        size_cat = "10M<n<100M"

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
- {size_cat}
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
    lines.append(f"- **Texte brut** : {raw_text_bytes / 1_000_000:.1f} MB (UTF-8)")
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
        sorted_sources = sorted(by_source.items(), key=lambda kv: -kv[1]["examples"])
        lines += [
            "## Répartition par source",
            "",
            f"> {len(sorted_sources)} sources.",
            "",
            "| # | source | exemples | tokens |",
            "|---:|---|---:|---:|",
        ]
        for i, (s, d) in enumerate(sorted_sources, 1):
            lines.append(f"| {i} | `{s}` | {d['examples']:,} | {d.get('tokens', 0):,} |")
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
