"""Borzoi track metadata loader.

`targets_human.txt.gz` columns:
    (unnamed index), identifier, file, clip, clip_soft, scale, sum_stat,
    strand_pair, description

Like Enformer, the ENCODE accession is embedded in the `file` path.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def load_borzoi_tracks(path: str | Path) -> pl.DataFrame:
    """Load Borzoi track metadata with `accession` + `assay` columns added."""
    df = pl.read_csv(path, separator="\t")
    # The first column is unnamed; polars usually labels it "" or "column_1".
    if "" in df.columns:
        df = df.rename({"": "track_index"})
    accession = pl.col("file").str.extract(r"(ENCSR[0-9A-Z]{6})", 1).alias("accession")
    assay = pl.col("description").str.split(":").list.get(0).alias("assay")
    return df.with_columns(accession, assay)
