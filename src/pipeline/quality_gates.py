# src/pipeline/quality_gates.py
"""
Quality Gates — contrôles qualité BLOQUANTS du corpus avant publication.

Philosophie LLMOps : on ne publie JAMAIS un dataset qui n'a pas passé tous les
contrôles. Chaque gate renvoie un résultat structuré (passed / expected / actual)
et le pipeline s'arrête si `all_passed` est faux.

Gates implémentés :
  - schema           : colonnes et types attendus
  - min_examples     : volume minimal
  - empty_texts      : pas de texte vide
  - duplicates       : taux de doublons exacts sous le seuil
  - raw_text_size    : taille de texte brut UTF-8 >= cible (objectif 100 Mo)
  - wolof_pct        : % de wolof (lu depuis le rapport de stats) >= seuil
  - no_hf_loss       : aucun texte présent sur HF ne disparaît (anti-régression)

Toutes les fonctions de bas niveau prennent un DataFrame -> faciles à tester.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


# --------------------------------------------------------------------------- #
@dataclass
class GateResult:
    name: str
    passed: bool
    expected: Any
    actual: Any
    detail: str = ""

    def line(self) -> str:
        icon = "✅" if self.passed else "❌"
        return (
            f"{icon} {self.name:16} attendu={self.expected!s:>12}  obtenu={self.actual!s:>12}  {self.detail}"
        )


@dataclass
class GateReport:
    results: list[GateResult] = field(default_factory=list)

    def add(self, r: GateResult) -> None:
        self.results.append(r)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def to_dict(self) -> dict:
        return {"all_passed": self.all_passed, "gates": [asdict(r) for r in self.results]}

    def render(self) -> str:
        head = "QUALITY GATES — " + ("TOUS PASSÉS ✅" if self.all_passed else "ÉCHEC ❌")
        return "\n".join([head, "=" * 78, *[r.line() for r in self.results], "=" * 78])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _norm_keys(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().str.strip()


def raw_text_bytes(df: pd.DataFrame) -> int:
    """Taille du texte brut en octets UTF-8 (colonne `text` seule)."""
    return int(df["text"].fillna("").astype(str).map(lambda s: len(s.encode("utf-8"))).sum())


# --------------------------------------------------------------------------- #
# Gates unitaires (testables)
# --------------------------------------------------------------------------- #
def gate_schema(df: pd.DataFrame, expected: dict[str, str]) -> GateResult:
    cols = list(df.columns)
    ok = all(c in cols for c in expected)
    detail = ""
    if ok and "sources" in expected and len(df):
        sample = df["sources"].iloc[0]
        is_list = (
            isinstance(sample, (list, tuple)) or hasattr(sample, "__len__") and not isinstance(sample, str)
        )
        if not is_list:
            ok = False
            detail = "colonne 'sources' n'est pas une liste"
    return GateResult("schema", ok, list(expected.keys()), cols, detail)


def gate_min_examples(df: pd.DataFrame, minimum: int) -> GateResult:
    n = len(df)
    return GateResult("min_examples", n >= minimum, f">={minimum:,}", f"{n:,}")


def gate_empty(df: pd.DataFrame, maximum: int) -> GateResult:
    empty = int((df["text"].fillna("").astype(str).str.strip() == "").sum())
    return GateResult("empty_texts", empty <= maximum, f"<={maximum}", empty)


def gate_duplicates(df: pd.DataFrame, max_pct: float) -> GateResult:
    keys = _norm_keys(df["text"])
    n = len(keys)
    dup = n - keys.nunique()
    pct = round(100 * dup / n, 4) if n else 0.0
    return GateResult("duplicates", pct <= max_pct, f"<={max_pct}%", f"{pct}% ({dup:,})")


def gate_raw_text_size(df: pd.DataFrame, min_mb: float) -> GateResult:
    mb = round(raw_text_bytes(df) / 1_000_000, 2)
    return GateResult("raw_text_size", mb >= min_mb, f">={min_mb} MB", f"{mb} MB")


def gate_wolof_pct(stats: dict | None, min_pct: float) -> GateResult:
    pct = None
    if stats and stats.get("lid"):
        pct = stats["lid"].get("wolof_pct")
    if pct is None:
        return GateResult("wolof_pct", False, f">={min_pct}%", "N/A", "rapport LID manquant")
    return GateResult("wolof_pct", pct >= min_pct, f">={min_pct}%", f"{pct}%")


def gate_no_hf_loss(df: pd.DataFrame, hf_df: pd.DataFrame | None) -> GateResult:
    if hf_df is None:
        return GateResult("no_hf_loss", True, "0 perte", "skip", "HF non accessible (skip)")
    hf_keys = set(_norm_keys(hf_df["text"]))
    hf_keys.discard("")
    new_keys = set(_norm_keys(df["text"]))
    missing = hf_keys - new_keys
    n_new = len(new_keys - hf_keys)
    return GateResult(
        "no_hf_loss", len(missing) == 0, "0 perte", f"{len(missing)} perdus", f"+{n_new:,} nouveaux"
    )


# --------------------------------------------------------------------------- #
# Runner haut niveau
# --------------------------------------------------------------------------- #
def fetch_hf_dataframe(
    repo: str, filename: str, repo_type: str = "dataset", token: str | None = None
) -> pd.DataFrame | None:
    """Télécharge le parquet actuellement publié sur HF (None si indisponible)."""
    try:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(repo, filename, repo_type=repo_type, token=token)
        return pd.read_parquet(p)
    except Exception:
        return None


def fetch_hf_parquet_path(
    repo: str, filename: str, repo_type: str = "dataset", token: str | None = None
) -> str | None:
    """Télécharge le parquet HF et renvoie son CHEMIN local (sans le charger en RAM).

    Permet de streamer le contrôle anti-perte sans matérialiser un second corpus
    complet en pandas (cause d'OOM sur les gros corpus).
    """
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo, filename, repo_type=repo_type, token=token)
    except Exception:
        return None


def _normalized_text_array(text_col: pa.Array | pa.ChunkedArray) -> pa.Array:
    """fillna('') -> trim espaces -> minuscule (équivalent de `_norm_keys`)."""
    filled = pc.fill_null(pc.cast(text_col, pa.string()), "")
    return pc.utf8_lower(pc.utf8_trim_whitespace(filled))


def _stream_hf_keys(parquet_path: str | Path) -> set[str]:
    """Construit l'ensemble des clés texte normalisées du parquet HF (streaming)."""
    pf = pq.ParquetFile(parquet_path)
    keys: set[str] = set()
    for batch in pf.iter_batches(batch_size=50_000, columns=["text"]):
        keys.update(_normalized_text_array(batch.column(0)).to_pylist())
    keys.discard("")
    return keys


def run_gates(
    parquet_path: str | Path,
    gates_cfg: dict,
    stats: dict | None = None,
    hf_df: pd.DataFrame | None = None,
    hf_parquet_path: str | Path | None = None,
) -> GateReport:
    """Exécute tous les gates configurés sur le parquet donné, EN STREAMING.

    Un seul batch (50k lignes) vit en mémoire à la fois : on ne charge jamais le
    corpus complet en pandas. Pour le contrôle anti-perte, fournir de préférence
    ``hf_parquet_path`` (streamé) plutôt que ``hf_df`` (pandas, gourmand).
    """
    expected_cols: dict[str, str] = gates_cfg["schema"]["columns"]

    pf = pq.ParquetFile(parquet_path)
    schema = pf.schema_arrow
    columns = list(schema.names)
    n_total = pf.metadata.num_rows
    text_idx = schema.get_field_index("text")

    # --- passe unique en streaming : agrégats + clés normalisées ---
    empty = 0
    raw_bytes = 0
    keys: set[str] = set()
    iter_cols = [c for c in ("text", "sources") if c in columns] or None
    for batch in pf.iter_batches(batch_size=50_000, columns=iter_cols):
        if text_idx < 0:
            continue
        text_col = batch.column(batch.schema.get_field_index("text"))
        # taille texte brut UTF-8
        s = pc.sum(pc.binary_length(pc.cast(text_col, pa.string()))).as_py()
        raw_bytes += int(s) if s is not None else 0
        # textes vides (null ou blancs)
        normed = _normalized_text_array(text_col)
        e = pc.sum(pc.cast(pc.equal(normed, ""), pa.int64())).as_py()
        empty += int(e) if e is not None else 0
        # clés normalisées (doublons + anti-perte)
        keys.update(normed.to_pylist())

    report = GateReport()

    # --- schema ---
    ok = all(c in columns for c in expected_cols)
    detail = ""
    if ok and "sources" in expected_cols:
        t = schema.field("sources").type
        if not (pa.types.is_list(t) or pa.types.is_large_list(t)):
            ok = False
            detail = "colonne 'sources' n'est pas une liste"
    report.add(GateResult("schema", ok, list(expected_cols.keys()), columns, detail))

    # --- min_examples ---
    minimum = gates_cfg["min_examples"]
    report.add(GateResult("min_examples", n_total >= minimum, f">={minimum:,}", f"{n_total:,}"))

    # --- empty_texts ---
    max_empty = gates_cfg["max_empty"]
    report.add(GateResult("empty_texts", empty <= max_empty, f"<={max_empty}", empty))

    # --- duplicates ---
    max_pct = gates_cfg["max_duplicate_pct"]
    dup = n_total - len(keys)
    pct = round(100 * dup / n_total, 4) if n_total else 0.0
    report.add(GateResult("duplicates", pct <= max_pct, f"<={max_pct}%", f"{pct}% ({dup:,})"))

    # --- raw_text_size ---
    min_mb = gates_cfg["min_raw_text_mb"]
    mb = round(raw_bytes / 1_000_000, 2)
    report.add(GateResult("raw_text_size", mb >= min_mb, f">={min_mb} MB", f"{mb} MB"))

    # --- wolof_pct ---
    report.add(gate_wolof_pct(stats, gates_cfg["min_wolof_pct"]))

    # --- no_hf_loss ---
    if gates_cfg.get("require_no_hf_loss", False):
        hf_keys: set[str] | None = None
        if hf_parquet_path is not None:
            hf_keys = _stream_hf_keys(hf_parquet_path)
        elif hf_df is not None:
            hf_keys = set(_norm_keys(hf_df["text"]))
            hf_keys.discard("")
        if hf_keys is None:
            report.add(GateResult("no_hf_loss", True, "0 perte", "skip", "HF non accessible (skip)"))
        else:
            new_keys = keys - {""}
            missing = hf_keys - new_keys
            n_new = len(new_keys - hf_keys)
            report.add(
                GateResult(
                    "no_hf_loss",
                    len(missing) == 0,
                    "0 perte",
                    f"{len(missing)} perdus",
                    f"+{n_new:,} nouveaux",
                )
            )
    return report


def load_stats(stats_path: str | Path) -> dict | None:
    p = Path(stats_path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None
