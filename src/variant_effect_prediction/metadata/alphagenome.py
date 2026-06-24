"""AlphaGenome track metadata loader.

Reads the bundled `alphagenome-track-metadata.parquet` (17 columns incl.
`track_index`, `track_name`, `output_type`, `biosample_name`, `assay_title`, …) and,
to map to ENCODE experiment accessions, the bundled
`alphagenome-supplemental-table-2.parquet` (its comma-separated `Experiment accession`
column is split + exploded for row-per-accession lookup). See `metadata/data/README.md`.

IMPORTANT: `track_index` is NOT unique on its own — it is only unique *within an
`output_type`* (the index restarts per head: there's an ATAC track 0, a DNase track 0,
etc.). Joining on `track_index` alone produces a cross-product across heads. We
therefore join on `(track_index, output_type)`, normalizing case (the track-metadata
stores lowercase `dnase`, the supplement uppercase `DNASE`). The returned `track_index`
is the per-head index that the AlphaGenome wrapper expects when scoring that head.

These loaders take no arguments — they always use the files we ship.
"""

from __future__ import annotations

import polars as pl

from variant_effect_prediction.metadata._data import (
    ALPHAGENOME_SUPP,
    ALPHAGENOME_TRACKS,
    read_packaged_parquet,
)


def load_alphagenome_tracks() -> pl.DataFrame:
    """Load the raw bundled AlphaGenome track metadata."""
    return read_packaged_parquet(ALPHAGENOME_TRACKS)


def load_alphagenome_accession_map(
    output_types: tuple[str, ...] = ("ATAC", "DNASE"),
) -> pl.DataFrame:
    """Build a (track_index, accession, biosample_name, output_type) lookup.

    Joins the bundled track metadata with the bundled supplementary table on the
    composite key `(track_index, output_type)` (since `track_index` is only unique per
    head). The supplement's comma-separated `Experiment accession` column is split and
    exploded for row-per-accession lookup. Filters to `organism == 'human'` and the
    requested `output_type` set. `output_type` in the returned frame is lowercase
    (matching the track metadata / the AlphaGenome head names like 'atac', 'dnase').
    """
    output_types_lc = [ot.lower() for ot in output_types]

    parquet = load_alphagenome_tracks().with_columns(
        pl.col("output_type").str.to_lowercase()
    )

    supp = read_packaged_parquet(ALPHAGENOME_SUPP)
    supp_filtered = (
        supp.filter(pl.col("organism") == "human")
        .with_columns(pl.col("output_type").str.to_lowercase())
        .filter(pl.col("output_type").is_in(output_types_lc))
        .with_columns(pl.col("Experiment accession").str.split(","))
        .explode("Experiment accession")
        .rename({"Experiment accession": "accession"})
        .select(["track_index", "output_type", "accession", "Assay title"])
        .unique()
    )

    # Composite key: track_index is only unique within an output_type.
    joined = parquet.join(
        supp_filtered,
        on=["track_index", "output_type"],
        how="inner",
    )
    return joined.select(
        pl.col("track_index"),
        pl.col("accession"),
        pl.col("biosample_name"),
        pl.col("output_type"),
        pl.col("Assay title").alias("assay_title"),
    )
