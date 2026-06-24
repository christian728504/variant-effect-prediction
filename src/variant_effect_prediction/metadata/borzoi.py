"""Borzoi track metadata loader.

Reads the bundled `borzoi-track-metadata.parquet` (see `metadata/data/README.md`),
which carries the upstream `targets_human.txt` columns:
    (unnamed index), identifier, file, clip, clip_soft, scale, sum_stat,
    strand_pair, description

Like Enformer, the ENCODE accession is embedded in the `file` path. This loader
takes no arguments — it always uses the file we ship.
"""

from __future__ import annotations

import polars as pl

from variant_effect_prediction.metadata._data import (
    BORZOI_TRACKS,
    read_packaged_parquet,
)


def load_borzoi_tracks() -> pl.DataFrame:
    """Load the bundled Borzoi track metadata with `track_index` + `accession` + `assay`."""
    df = read_packaged_parquet(BORZOI_TRACKS)
    # The upstream first column is unnamed; it carries the per-track index.
    if "" in df.columns:
        df = df.rename({"": "track_index"})
    accession = pl.col("file").str.extract(r"(ENCSR[0-9A-Z]{6})", 1).alias("accession")
    assay = pl.col("description").str.split(":").list.get(0).alias("assay")
    return df.with_columns(accession, assay)
