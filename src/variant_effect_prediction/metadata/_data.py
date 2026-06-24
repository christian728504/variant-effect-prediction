"""Access to the track-metadata files bundled as package data.

The four parquet files under `metadata/data/` (enformer / borzoi / alphagenome
track metadata + the AlphaGenome supplementary table 2) ship inside the wheel, so
the loaders work out-of-the-box without the caller pointing at the repo. See
`metadata/data/README.md` for each file's provenance.
"""

from __future__ import annotations

from importlib.resources import as_file, files

import polars as pl

# Bundled-file names (parquet). Keep in sync with metadata/data/.
ENFORMER_TARGETS = "enformer-targets-human.parquet"
BORZOI_TRACKS = "borzoi-track-metadata.parquet"
ALPHAGENOME_TRACKS = "alphagenome-track-metadata.parquet"
ALPHAGENOME_SUPP = "alphagenome-supplemental-table-2.parquet"

# `__package__` here is "variant_effect_prediction.metadata".
_DATA = files(__package__).joinpath("data")


def read_packaged_parquet(filename: str) -> pl.DataFrame:
    """Read one of the bundled parquet files by name (robust to zipped installs)."""
    with as_file(_DATA.joinpath(filename)) as p:
        return pl.read_parquet(p)
