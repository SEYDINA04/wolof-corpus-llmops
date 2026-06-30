#!/usr/bin/env python3
"""Découpe le corpus unifié en exemples courts (max 2 phrases).

Règles (voir discussion projet) :
  - Phrases délimitées par . ? ! …  + sauts de ligne ; la ponctuation finale
    reste collée à la phrase.
  - Les exemples contenant un bloc de code (``` ``` ```) sont gardés INTACTS
    (les couper sur la ponctuation casserait le code).
  - Regroupement par paires : au plus 2 phrases par exemple de sortie.
  - Nettoyage : dédup (texte normalisé) + suppression des fragments < MIN_LEN.
  - Traitement EN STREAMING (mémoire constante) : un set de hash 64 bits pour la
    dédup, écriture par batches via ParquetWriter -> jamais tout le corpus en RAM.

Usage :
    python split_corpus.py [SRC_PARQUET] [OUT_PARQUET]
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# --- Paramètres ----------------------------------------------------------- #
SRC_DEFAULT = "wolof_unified_corpus/data/train-00000-of-00001.parquet"
OUT_DEFAULT = "wolof_split_corpus/data/train-00000-of-00001.parquet"
MAX_SENTENCES = 2  # phrases max par exemple de sortie
MIN_LEN = 10  # longueur mini (caractères) d'un fragment conservé
BATCH = 50_000

# Coupe APRÈS une ponctuation finale (. ! ? …) suivie d'espace(s).
# Le lookbehind garde la ponctuation avec la phrase de gauche ; comme on exige
# un espace après, "3.14" ou "1.000" ne sont pas coupés (pas d'espace).
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def has_code(text: str) -> bool:
    """Détecte un bloc de code (triple backticks) -> exemple gardé intact."""
    return "```" in text


def split_sentences(text: str) -> list[str]:
    """Segmente en phrases : d'abord par lignes, puis par ponctuation finale."""
    sentences: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for s in _SENT_SPLIT.split(line):
            s = s.strip()
            if s:
                sentences.append(s)
    return sentences


def make_chunks(text: str) -> list[str]:
    """Retourne les morceaux de sortie pour un texte donné."""
    if has_code(text):
        t = text.strip()
        return [t] if t else []
    sents = split_sentences(text)
    if not sents:
        t = text.strip()
        return [t] if t else []
    return [
        " ".join(sents[i : i + MAX_SENTENCES])
        for i in range(0, len(sents), MAX_SENTENCES)
    ]


def norm_key(s: str) -> str:
    """Clé de dédup : minuscule + espaces normalisés."""
    return " ".join(s.lower().split())


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else SRC_DEFAULT)
    out = Path(sys.argv[2] if len(sys.argv) > 2 else OUT_DEFAULT)
    out.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema(
        [("text", pa.string()), ("sources", pa.list_(pa.string()))]
    )
    writer = pq.ParquetWriter(out, schema, compression="snappy")

    seen: set[int] = set()
    buf_text: list[str] = []
    buf_src: list[list[str] | None] = []

    n_in = 0
    n_out = 0
    n_dup = 0
    n_short = 0
    n_code = 0
    raw_bytes = 0
    t0 = time.time()

    def flush() -> None:
        if buf_text:
            writer.write_table(
                pa.table({"text": buf_text, "sources": buf_src}, schema=schema)
            )
            buf_text.clear()
            buf_src.clear()

    pf = pq.ParquetFile(src)
    total = pf.metadata.num_rows
    for batch in pf.iter_batches(batch_size=BATCH, columns=["text", "sources"]):
        d = batch.to_pydict()
        for txt, src_list in zip(d["text"], d["sources"]):
            n_in += 1
            if not txt:
                continue
            code = has_code(txt)
            if code:
                n_code += 1
            for ch in make_chunks(txt):
                if len(ch) < MIN_LEN:
                    n_short += 1
                    continue
                h = hash(norm_key(ch))
                if h in seen:
                    n_dup += 1
                    continue
                seen.add(h)
                buf_text.append(ch)
                buf_src.append(src_list)
                raw_bytes += len(ch.encode("utf-8"))
                n_out += 1
                if len(buf_text) >= BATCH:
                    flush()
        print(
            f"\r  lu {n_in:,}/{total:,}  ->  sortie {n_out:,}",
            end="",
            flush=True,
        )

    flush()
    writer.close()
    dt = time.time() - t0
    mb = raw_bytes / 1_000_000
    disk = out.stat().st_size / 1_000_000
    print(f"\n\n=== SPLIT TERMINÉ en {dt:.1f}s ===")
    print(f"  exemples en entrée   : {n_in:,}")
    print(f"  exemples en sortie   : {n_out:,}")
    print(f"  exemples 'code' intacts : {n_code:,}")
    print(f"  fragments < {MIN_LEN} car retirés : {n_short:,}")
    print(f"  doublons supprimés   : {n_dup:,}")
    print(f"  texte brut (UTF-8)   : {mb:.2f} MB")
    print(f"  parquet sur disque   : {disk:.2f} MB (SNAPPY)")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
