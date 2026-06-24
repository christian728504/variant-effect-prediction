"""Enformer track metadata loader.

Reads the bundled `enformer-targets-human.parquet` (see `metadata/data/README.md`),
which carries the upstream `targets_human.txt` columns:
    index, genome, identifier, file, clip, scale, sum_stat, description

The ENCODE experiment accession is embedded in the `file` path
(e.g. `.../encode/ENCSR000EIJ/summary/coverage.w5`). We extract it via regex.
This loader takes no arguments — it always uses the file we ship.
"""

from __future__ import annotations

import polars as pl

from variant_effect_prediction.metadata._data import (
    ENFORMER_TARGETS,
    read_packaged_parquet,
)


def load_enformer_tracks() -> pl.DataFrame:
    """Load the bundled Enformer track metadata with `accession` + `assay` added."""
    df = read_packaged_parquet(ENFORMER_TARGETS)
    accession = pl.col("file").str.extract(r"(ENCSR[0-9A-Z]{6})", 1).alias("accession")
    assay = pl.col("description").str.split(":").list.get(0).alias("assay")
    return df.with_columns(accession, assay)
