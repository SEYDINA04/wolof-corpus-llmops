# src/tests/test_quality_gates.py
"""
Tests unitaires des quality gates — rapides, sur DataFrames synthétiques.
Aucune dépendance réseau : exécutables en CI sur chaque push/PR.
"""

import pandas as pd
import pytest

from pipeline import quality_gates as qg


@pytest.fixture
def good_df():
    return pd.DataFrame({
        "text": ["man maa ngi fi", "naka nga def", "jërëjëf waay"],
        "sources": [["a"], ["b"], ["a", "b"]],
    })


def test_schema_ok(good_df):
    r = qg.gate_schema(good_df, {"text": "str", "sources": "list"})
    assert r.passed


def test_schema_missing_column(good_df):
    df = good_df.drop(columns=["sources"])
    r = qg.gate_schema(df, {"text": "str", "sources": "list"})
    assert not r.passed


def test_schema_sources_not_list():
    df = pd.DataFrame({"text": ["a b c"], "sources": ["x"]})  # str au lieu de liste
    r = qg.gate_schema(df, {"text": "str", "sources": "list"})
    assert not r.passed


def test_min_examples(good_df):
    assert qg.gate_min_examples(good_df, 3).passed
    assert not qg.gate_min_examples(good_df, 4).passed


def test_empty_texts():
    df = pd.DataFrame({"text": ["ok ok", "   ", ""], "sources": [["a"], ["a"], ["a"]]})
    assert qg.gate_empty(df, maximum=2).passed
    assert not qg.gate_empty(df, maximum=1).passed


def test_duplicates():
    df = pd.DataFrame({
        "text": ["Bonjour", "bonjour", "autre"],   # 2 doublons (casse ignorée)
        "sources": [["a"], ["b"], ["c"]],
    })
    r0 = qg.gate_duplicates(df, max_pct=0.0)
    assert not r0.passed                # 33% de doublons
    r1 = qg.gate_duplicates(df, max_pct=50.0)
    assert r1.passed


def test_raw_text_size():
    df = pd.DataFrame({"text": ["x" * 1_000_000], "sources": [["a"]]})
    assert qg.gate_raw_text_size(df, min_mb=0.5).passed
    assert not qg.gate_raw_text_size(df, min_mb=2.0).passed


def test_wolof_pct():
    assert qg.gate_wolof_pct({"lid": {"wolof_pct": 95.0}}, 90.0).passed
    assert not qg.gate_wolof_pct({"lid": {"wolof_pct": 80.0}}, 90.0).passed
    assert not qg.gate_wolof_pct(None, 90.0).passed   # rapport manquant -> échec


def test_no_hf_loss():
    new = pd.DataFrame({"text": ["a a", "b b", "c c"], "sources": [["x"]] * 3})
    hf_ok = pd.DataFrame({"text": ["a a", "b b"], "sources": [["x"]] * 2})
    assert qg.gate_no_hf_loss(new, hf_ok).passed          # tout HF est présent
    hf_loss = pd.DataFrame({"text": ["a a", "ZZZ"], "sources": [["x"]] * 2})
    assert not qg.gate_no_hf_loss(new, hf_loss).passed    # 'ZZZ' perdu
    assert qg.gate_no_hf_loss(new, None).passed           # HF inaccessible -> skip


def test_report_aggregation(good_df, tmp_path):
    p = tmp_path / "c.parquet"
    good_df.to_parquet(p, index=False)
    cfg = {
        "schema": {"columns": {"text": "str", "sources": "list"}},
        "min_examples": 3, "max_empty": 0, "max_duplicate_pct": 0.0,
        "min_raw_text_mb": 0.0, "min_wolof_pct": 0.0, "require_no_hf_loss": False,
    }
    report = qg.run_gates(p, cfg, stats={"lid": {"wolof_pct": 100.0}})
    assert report.all_passed
    assert len(report.results) == 6
