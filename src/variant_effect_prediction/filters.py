"""var.isused filter builder.

We implement two of the three upstream sub-filters from the ChromBPNet paper:

    1. Peak-overlap: variant within ±summit_slop_bp of a top-N MACS2 peak summit.
    2. Significance threshold: -log10(pvalue) > significance_cutoff.

The third upstream filter (precomputed-Enformer-availability) is deliberately
omitted — we run Enformer ourselves and don't depend on
gs://dm-enformer/variant-scores/1000genomes/enformer.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def _read_peaks_bed(
    peak_bed_path: str | Path,
    top_n_peaks: int,
    summit_slop_bp: int,
) -> pl.DataFrame:
    """Read a narrowPeak / BED file and return summit windows.

    Expects narrowPeak schema: chrom, start, end, name, score, strand, signalValue,
    pValue, qValue, peak (relative summit offset). Top peaks are picked by `signalValue`.
    """
    df = pl.read_csv(
        peak_bed_path,
        separator="\t",
        has_header=False,
        new_columns=[
            "chrom",
            "start",
            "end",
            "name",
            "score",
            "strand",
            "signalValue",
            "pValue",
            "qValue",
            "peak",
        ],
        schema_overrides={
            "chrom": pl.Utf8,
            "start": pl.UInt32,
            "end": pl.UInt32,
            "signalValue": pl.Float64,
            "peak": pl.Int32,
        },
        truncate_ragged_lines=True,
    )
    summit_abs = (df["start"].cast(pl.Int64) + df["peak"].cast(pl.Int64)).cast(
        pl.UInt32
    )
    df = df.with_columns(summit_abs.alias("summit"))
    df = df.sort("signalValue", descending=True).head(top_n_peaks)
    return df.select(
        pl.col("chrom"),
        (pl.col("summit").cast(pl.Int64) - summit_slop_bp)
        .clip(lower_bound=0)
        .cast(pl.UInt32)
        .alias("start"),
        (pl.col("summit").cast(pl.Int64) + summit_slop_bp).cast(pl.UInt32).alias("end"),
    )


def _peak_overlap_mask(
    variants: pl.DataFrame,
    peaks: pl.DataFrame,
) -> pl.Series:
    """Return a Boolean Series the length of `variants` indicating overlap with any peak.

    Implemented as a left join_asof followed by an explicit interval predicate so we
    don't need polars-bio as a hard dependency. Variants are matched against the nearest
    peak start within the same chrom, then verified to lie inside the peak window.
    """
    variants = variants.with_row_index("__vidx")
    v_sorted = variants.sort(["chrom", "start"])
    p_sorted = peaks.sort(["chrom", "start"])

    joined = v_sorted.join_asof(
        p_sorted.rename({"start": "peak_start", "end": "peak_end"}),
        by="chrom",
        left_on="start",
        right_on="peak_start",
        strategy="backward",
    )

    overlap = (
        joined["peak_start"].is_not_null()
        & (joined["start"] >= joined["peak_start"])
        & (joined["end"] <= joined["peak_end"])
    )
    joined = joined.with_columns(overlap.alias("__overlap"))

    return joined.sort("__vidx")["__overlap"].fill_null(False)


def build_isused(
    df: pl.DataFrame,
    *,
    peak_bed_path: str | Path,
    summit_slop_bp: int = 100,
    top_n_peaks: int = 50_000,
    significance_cutoff: int | None = None,
) -> pl.Series:
    """Compute the boolean isused mask from canonical-schema variants + a peak BED.

    `significance_cutoff` applies the dataset's significance rule
    `-log10(pvalue) > significance_cutoff` (e.g. 6 for European caQTLs). Variants
    with a null pvalue are treated as not significant. When omitted, only the
    peak-overlap sub-filter is applied.
    """
    peaks = _read_peaks_bed(peak_bed_path, top_n_peaks, summit_slop_bp)
    mask = _peak_overlap_mask(df.select(["chrom", "start", "end"]), peaks)

    if significance_cutoff is not None:
        sig_mask = (-df["pvalue"].log10() > significance_cutoff).fill_null(False)
        mask = mask & sig_mask

    return mask.alias("isused")
