"""VariantSet: a Polars DataFrame wrapper enforcing a strict canonical schema for
QTL / GWAS variants.

The canonical schema is:
    chrom        (Utf8)
    start        (UInt32)  0-based inclusive
    end          (UInt32)  0-based exclusive
    allele1      (Utf8)    A/C/G/T only  (the QTL "reference-like" allele)
    allele2      (Utf8)    A/C/G/T only  (the QTL "alternate" allele)
    effect_size  (Float64)
    pvalue       (Float64 | Null)
    isused       (Boolean)
    raw          (Struct)  packed original row

We use the variant-scorer `allele1`/`allele2` naming (not ref/alt): logFC is
log2(allele2 / allele1) and the genome is not required to match allele1.

QTL files have free-form column names, so callers map raw columns to canonical
names at construction via VariantSet.from_dataframe / from_tsv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


_VALID_BASES = frozenset({"A", "C", "G", "T"})


class VariantSetSchemaError(ValueError):
    """Raised when an input DataFrame does not match the canonical VariantSet schema."""


class VariantSet:
    """Strict-schema wrapper around a Polars DataFrame of variants."""

    CANONICAL_COLUMNS: tuple[str, ...] = (
        "chrom",
        "start",
        "end",
        "allele1",
        "allele2",
        "effect_size",
        "pvalue",
        "isused",
        "raw",
    )

    def __init__(self, df: pl.DataFrame, *, meta: dict[str, Any] | None = None) -> None:
        self._validate(df)
        self._df = df
        # Repr-only reporting metadata (name, source_path, filters_applied,
        # n_total). Never read by any scoring logic — see __repr__.
        self._meta: dict[str, Any] = dict(meta) if meta else {}

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        *,
        chrom_col: str,
        start_col: str,
        allele1_col: str,
        allele2_col: str,
        effect_size_col: str,
        end_col: str | None = None,
        pvalue_col: str | None = None,
        isused_col: str | None = None,
        apply_isused: bool = True,
        snvs_only: bool = True,
        peak_bed_path: str | Path | None = None,
        significance_cutoff: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> "VariantSet":
        """Build a VariantSet from a raw DataFrame plus a column mapping.

        `snvs_only` (default True) keeps only single-nucleotide variants — both
        allele1 and allele2 exactly one base — dropping indels. The sequence-to-
        function scorers center a fixed window on a single variant position, so
        indel length changes confound the eval window; SNV-only is the sane
        default for scoring. Set False to retain indels (e.g. for inspection).
        """

        required_inputs = {
            chrom_col,
            start_col,
            allele1_col,
            allele2_col,
            effect_size_col,
        }
        missing = required_inputs - set(df.columns)
        if missing:
            raise VariantSetSchemaError(
                f"Input DataFrame is missing required columns: {sorted(missing)}"
            )

        if snvs_only:
            # Filter the raw frame up front so raw/isused/etc. stay row-aligned.
            df = df.filter(
                (pl.col(allele1_col).cast(pl.Utf8).str.len_bytes() == 1)
                & (pl.col(allele2_col).cast(pl.Utf8).str.len_bytes() == 1)
            )

        raw_cols = [c for c in df.columns if c is not None]
        packed_raw = df.select(pl.struct(raw_cols).alias("raw"))

        out = df.select(
            pl.col(chrom_col).cast(pl.Utf8).alias("chrom"),
            pl.col(start_col).cast(pl.UInt32).alias("start"),
            pl.col(allele1_col).cast(pl.Utf8).str.to_uppercase().alias("allele1"),
            pl.col(allele2_col).cast(pl.Utf8).str.to_uppercase().alias("allele2"),
            pl.col(effect_size_col).cast(pl.Float64).alias("effect_size"),
        )

        if end_col is not None:
            out = out.with_columns(pl.col(end_col).cast(pl.UInt32).alias("end"))
        else:
            out = out.with_columns(
                (
                    pl.col("start")
                    + pl.max_horizontal(
                        pl.col("allele1").str.len_bytes(),
                        pl.col("allele2").str.len_bytes(),
                    ).cast(pl.UInt32)
                ).alias("end")
            )

        if pvalue_col is not None:
            if pvalue_col not in df.columns:
                raise VariantSetSchemaError(
                    f"pvalue_col={pvalue_col!r} not present in input DataFrame"
                )
            pvalue_series = df.select(
                pl.col(pvalue_col).cast(pl.Float64).alias("pvalue")
            )
            out = pl.concat([out, pvalue_series], how="horizontal")
        else:
            out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("pvalue"))

        if isused_col is not None:
            if isused_col not in df.columns:
                raise VariantSetSchemaError(
                    f"isused_col={isused_col!r} not present in input DataFrame"
                )
            isused_series = df.select(
                pl.col(isused_col).cast(pl.Boolean).alias("isused")
            )
            out = pl.concat([out, isused_series], how="horizontal")
        elif apply_isused:
            # Defer the heavy filter to a separate module so callers without peak
            # files can still construct VariantSets via isused_col / passthrough.
            from variant_effect_prediction.filters import build_isused

            if peak_bed_path is None:
                raise VariantSetSchemaError(
                    "apply_isused=True requires peak_bed_path (or pass isused_col / "
                    "apply_isused=False)"
                )
            isused = build_isused(
                out,
                peak_bed_path=peak_bed_path,
                significance_cutoff=significance_cutoff,
            )
            out = out.with_columns(isused.alias("isused"))
        else:
            out = out.with_columns(pl.lit(True).alias("isused"))

        out = pl.concat([out, packed_raw], how="horizontal")
        out = out.select(list(cls.CANONICAL_COLUMNS))

        return cls(out, meta=meta)

    @classmethod
    def from_tsv(cls, path: str | Path, **kwargs: Any) -> "VariantSet":
        df = pl.read_csv(path, separator="\t")
        return cls.from_dataframe(df, **kwargs)

    @classmethod
    def _validate(cls, df: pl.DataFrame) -> None:
        missing = [c for c in cls.CANONICAL_COLUMNS if c not in df.columns]
        if missing:
            raise VariantSetSchemaError(
                f"DataFrame missing canonical columns: {missing}. Got: {df.columns}"
            )

        expected_dtypes = {
            "chrom": pl.Utf8,
            "start": pl.UInt32,
            "end": pl.UInt32,
            "allele1": pl.Utf8,
            "allele2": pl.Utf8,
            "effect_size": pl.Float64,
            "pvalue": pl.Float64,
            "isused": pl.Boolean,
        }
        bad = [
            (c, df.schema[c], dt)
            for c, dt in expected_dtypes.items()
            if df.schema[c] != dt
        ]
        if bad:
            raise VariantSetSchemaError(
                "Canonical schema dtype mismatch (col, actual, expected): " + str(bad)
            )

        if not isinstance(df.schema["raw"], pl.Struct):
            raise VariantSetSchemaError(
                f"raw column must be a Struct, got {df.schema['raw']}"
            )

        bad_bases = df.filter(
            ~pl.col("allele1")
            .str.contains_any(list(_VALID_BASES))
            .or_(pl.col("allele2").str.contains_any(list(_VALID_BASES)))
        )
        if bad_bases.height > 0:
            raise VariantSetSchemaError(
                f"{bad_bases.height} rows have allele1/allele2 outside {{A,C,G,T}}"
            )

    @property
    def df(self) -> pl.DataFrame:
        return self._df

    def __len__(self) -> int:
        return self._df.height

    @property
    def meta(self) -> dict[str, Any]:
        """Repr-only reporting metadata (see from_dataframe(meta=...))."""
        return self._meta

    def __repr__(self) -> str:
        n_used = int(self._df["isused"].sum())
        m = self._meta
        head = f"VariantSet(n={len(self)}, n_used={n_used}"
        if m.get("name"):
            head = f"VariantSet(name={m['name']!r}, n={len(self)}, n_used={n_used}"
        lines = [head + ")"]
        if m.get("source_path"):
            lines.append(f"  source: {m['source_path']}")
        if m.get("n_total") is not None:
            lines.append(f"  n_total (pre-filter): {m['n_total']}")
        if m.get("filters_applied"):
            lines.append(f"  filters: {m['filters_applied']}")
        lines.append("  schema:")
        for col, dt in self._df.schema.items():
            lines.append(f"    {col}: {dt}")
        return "\n".join(lines)

    def used(self) -> "VariantSet":
        """Return a new VariantSet filtered to rows with isused == True."""
        return VariantSet(self._df.filter(pl.col("isused")), meta=self._meta)

    def head(self, n: int = 5) -> "VariantSet":
        return VariantSet(self._df.head(n), meta=self._meta)

    def sample(self, n: int, seed: int = 0) -> "VariantSet":
        return VariantSet(self._df.sample(n=n, seed=seed), meta=self._meta)
