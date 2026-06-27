# src/tests/test_datacard.py
"""Test du générateur de data card."""

import pandas as pd

from pipeline import datacard


def test_build_readme(tmp_path):
    df = pd.DataFrame({
        "text": ["man maa ngi fi", "naka nga def"],
        "sources": [["galsenai/x"], ["y"]],
    })
    p = tmp_path / "c.parquet"
    df.to_parquet(p, index=False)
    stats = {
        "total_tokens": 7,
        "lid": {"wolof_pct": 95.0},
        "by_source": {"galsenai/x": {"examples": 1, "tokens": 4}},
    }
    md = datacard.build_readme(p, stats)
    assert md.startswith("---")            # front-matter YAML
    assert "num_examples: 2" in md
    assert "language:\n- wo" in md
    assert "galsenai/x" in md
    assert "95.0%" in md
