"""Enformer track metadata loader.

The `targets_human.txt` file ships with enformer-pytorch and has columns:
    index, genome, identifier, file, clip, scale, sum_stat, description

The ENCODE experiment accession is embedded in the `file` path
(e.g. `.../encode/ENCSR000EIJ/summary/coverage.w5`). We extract it via regex.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def load_enformer_tracks(path: str | Path) -> pl.DataFrame:
    """Load Enformer track metadata with an `accession` column added when present."""
    df = pl.read_csv(path, separator="\t")
    accession = pl.col("file").str.extract(r"(ENCSR[0-9A-Z]{6})", 1).alias("accession")
    assay = pl.col("description").str.split(":").list.get(0).alias("assay")
    return df.with_columns(accession, assay)
