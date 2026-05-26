"""AlphaGenome track metadata loader.

`track_metadata_human.parquet` has 17 columns including `track_index`,
`track_name`, `output_type`, `biosample_name`, `assay_title`, etc.

To map to ENCODE experiment accessions we join the parquet with
`Suppl Table 2 Track metadata (f.tsv` (in `alphagenome_supplementary_tables.d/`)
which has a comma-separated `Experiment accession` column that must be split and
exploded for row-per-accession lookup.

IMPORTANT: `track_index` is NOT unique on its own — it is only unique *within an
`output_type`* (the index restarts per head: there's an ATAC track 0, a DNase
track 0, etc.). Joining on `track_index` alone produces a cross-product across
heads. We therefore join on `(track_index, output_type)`, normalizing case
(parquet stores lowercase `dnase`, the supplement stores uppercase `DNASE`). The
returned `track_index` is the per-head index that the AlphaGenome wrapper expects
when scoring that head.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def load_alphagenome_tracks(parquet_path: str | Path) -> pl.DataFrame:
    """Load the raw AlphaGenome track metadata parquet."""
    return pl.read_parquet(parquet_path)


def load_alphagenome_accession_map(
    parquet_path: str | Path,
    supplementary_tsv_path: str | Path,
    output_types: tuple[str, ...] = ("ATAC", "DNASE"),
) -> pl.DataFrame:
    """Build a (track_index, accession, biosample_name, output_type) lookup.

    Joins on the composite key `(track_index, output_type)` since `track_index`
    is only unique per head. The supplementary table's comma-separated
    `Experiment accession` column is split and exploded for row-per-accession
    lookup. Filters to `organism == 'human'` and the requested `output_type` set.
    `output_type` in the returned frame is lowercase (matching the parquet / the
    AlphaGenome head names like 'atac', 'dnase').
    """
    output_types_lc = [ot.lower() for ot in output_types]

    parquet = load_alphagenome_tracks(parquet_path).with_columns(
        pl.col("output_type").str.to_lowercase()
    )

    supp = pl.read_csv(supplementary_tsv_path, separator="\t")
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
