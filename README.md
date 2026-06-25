# variant-effect-prediction

A Python library for scoring non-coding genetic variants with sequence-to-function deep learning models. It puts five published models — **ChromBPNet**, **Cherimoya**, **AlphaGenome**, **Borzoi**, and **Enformer** — behind one scoring API: feed it a table of variants and a reference genome, get back each model's predicted allelic effect (logFC) and profile shift (JSD).

The design goal is *parity*: predicted effects are computed to match the reference ChromBPNet variant-scorer convention exactly, so scores are comparable across models.

## Features

- **One API for five models.** Every model is a `VariantScorer` subclass implementing a single `_predict_alleles` method; the base class handles sequence extraction, batching, logFC, and JSD. Calling code only ever touches `scorer.score(variant_set)`.
- **Strict, self-validating variant tables.** `VariantSet` wraps a Polars DataFrame with an enforced canonical schema and maps free-form QTL/GWAS column names onto it at construction, dropping indels and applying `isused` masks.
- **Reference-implementation parity.** logFC is computed in log space (`log2(allele2 / allele1)`) and JSD via mean-subtracted softmax / sum normalization to match the published ChromBPNet `variant_effect` scoring code. Allele extraction follows the variant-scorer convention (no ref-base check, boundary N-padding).
- **Build-agnostic reference access.** `RefGenome` wraps any pysam/faidx-indexed FASTA (GRCh37, GRCh38, …); the build is just whichever FASTA you point it at.
- **Bundled track metadata.** Enformer / Borzoi / AlphaGenome track tables and ENCODE accession maps ship as package-data parquet, so resolving a many-tracks model's track index needs no external downloads.

## Installation

```bash
git clone git@github.com:christian728504/variant-effect-prediction.git
cd variant-effect-prediction
uv sync
```

Requires Python ≥ 3.12 and a CUDA GPU (scorers default to `device="cuda"`). The model packages (`bpnet-lite`, `cherimoya`, `alphagenome-pytorch`, `borzoi-pytorch`, `enformer-pytorch`) are pinned to specific git revisions in `pyproject.toml` and installed by `uv sync`.

## Quick start

```python
from variant_effect_prediction import RefGenome, VariantSet, FoldedModelWeights
from variant_effect_prediction.scorers import ChromBPNetVariantScorer

ref = RefGenome("GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz")

# Map free-form QTL columns onto the canonical schema.
vs = VariantSet.from_tsv(
    "my_qtls.tsv",
    chrom_col="var.chr",
    start_col="start0",          # 0-based variant position
    allele1_col="var.allele1",   # QTL reference-like allele
    allele2_col="var.allele2",   # QTL alternate allele
    effect_size_col="obs.beta",
    apply_isused=False,
)

weights = FoldedModelWeights.from_dir("weights/chrombpnet/ENCSR637XSC")
scorer = ChromBPNetVariantScorer(folded_weights=weights, ref_genome=ref)

scored = scorer.score(vs)   # Polars DataFrame: logfc, jsd, per-allele predictions, …
```

## Core concepts

### `VariantSet` — the canonical variant table

`VariantSet` enforces a strict schema so every scorer receives identically-shaped input:

| Column        | Type            | Meaning                                          |
|---------------|-----------------|--------------------------------------------------|
| `chrom`       | Utf8            | Chromosome                                        |
| `start`/`end` | UInt32          | 0-based half-open variant interval               |
| `allele1`     | Utf8            | QTL reference-like allele (A/C/G/T)              |
| `allele2`     | Utf8            | QTL alternate allele (A/C/G/T)                   |
| `effect_size` | Float64         | Measured effect (e.g. QTL β, log2FC)             |
| `pvalue`      | Float64 \| Null | Optional significance                            |
| `isused`      | Boolean         | Whether the variant is scored                    |
| `raw`         | Struct          | The original row, packed (recover any extra col) |

QTL/GWAS files have free-form column names, so callers map them at construction:

```python
vs = VariantSet.from_dataframe(df, chrom_col=..., start_col=..., allele1_col=...,
                               allele2_col=..., effect_size_col=...)
```

`from_dataframe` / `from_tsv` accept optional `pvalue_col`, `isused_col`, `snvs_only` (default `True` — drops indels), and `apply_isused` (with a `peak_bed_path` + `significance_cutoff`) to build the `isused` mask from a peak BED. `vs.used()` returns the subset that will be scored; `vs.df` exposes the underlying Polars frame.

> [!NOTE]
> Allele naming follows the variant-scorer convention (`allele1`/`allele2`, **not** ref/alt): logFC is `log2(allele2 / allele1)` and the genome is **not** required to carry `allele1`.

### `RefGenome` — reference sequence access

`RefGenome(fasta_path)` wraps `pysam.FastaFile`. Its `extract_alleles(chrom, start, allele1, allele2, context_len)` returns the two fixed-length allele sequences centered on the variant — following the ChromBPNet variant-scorer convention (no check that the genome base equals `allele1`, indels length-normalized by trimming the downstream flank, chromosome-boundary positions N-padded rather than dropped). Scorers call this internally; the model's `context_len` is supplied automatically.

### `FoldedModelWeights` — 5-fold weight container

BPNet-like scorers (ChromBPNet, Cherimoya) ensemble five folds. `FoldedModelWeights.from_dir(dir, pattern="fold_{i}.torch")` locates the five per-fold torch state-dict files in a `<MODEL_NAME>/fold_{i}.torch` directory and validates they exist. Each fold file is a `{"config", "state_dict"}` payload that the scorer reconstructs.

Cherimoya already saves models in this format (pass `pattern="fold_{i}.final.torch"`). ChromBPNet weights ship as TensorFlow h5 (ENCODE `model.tar.gz` or `models/fold_N/chrombpnet_nobias.h5`); convert them once with `scripts/convert_chrombpnet_to_torch.py`:

```bash
python scripts/convert_chrombpnet_to_torch.py \
    --tar ENCSR637XSC/model.tar.gz --eid ENCSR637XSC --out weights/chrombpnet/ENCSR637XSC
```

### Scorers

All scorers subclass `VariantScorer` and share the `score(vs) -> pl.DataFrame` interface. The result frame carries the variant columns plus `allele1_pred`, `allele2_pred` (per-allele prediction arrays), `logfc` (`log2(allele2 / allele1)`), and `jsd` (signed Jensen–Shannon distance of the profiles, or null when the model exposes no profile).

| Scorer                       | Type                        | Context | Profile / JSD       |
|------------------------------|-----------------------------|---------|---------------------|
| `ChromBPNetVariantScorer`    | BPNet-like, 5-fold ensemble | 2114 bp | softmax-normalized  |
| `CherimoyaVariantScorer`     | BPNet-like, 5-fold ensemble | 2114 bp | softmax-normalized  |
| `AlphaGenomeVariantScorer`   | many-tracks                 | long    | sum-normalized      |
| `BorzoiVariantScorer`        | many-tracks                 | 524 kb  | sum-normalized      |
| `EnformerVariantScorer`      | many-tracks                 | long    | sum-normalized      |

The BPNet-like scorers take `folded_weights` + `ref_genome`. The many-tracks scorers take a loaded `model`, a `track_idx` (which output tracks to read), and `ref_genome`; track indices for an ENCODE accession can be resolved via the metadata loaders below. All accept `batch_size`, `device`, and `dtype`.

### Metadata loaders

`variant_effect_prediction.metadata` resolves many-tracks model outputs from bundled parquet:

```python
from variant_effect_prediction.metadata import (
    load_enformer_tracks, load_borzoi_tracks,
    load_alphagenome_tracks, load_alphagenome_accession_map,
)
```

These return Polars frames mapping ENCODE accessions to the track indices each model emits — feed the resulting index into a many-tracks scorer's `track_idx`.

## Package layout

```
src/variant_effect_prediction/
  variantset.py   VariantSet — canonical-schema variant table
  references.py   RefGenome — pysam-backed FASTA + allele extraction
  weights.py      FoldedModelWeights — 5-fold state-dict path container + from_dir loader
  scoring.py      logFC / JSD primitives (reference-parity)
  scorers/        VariantScorer ABC + per-model scorers
  wrappers/       tangermeme-compatible many-tracks model wrappers
  metadata/       track-metadata loaders + ENCODE accession maps (bundled parquet)
```

## Testing

```bash
uv run pytest
```

The suite covers the scoring primitives, `VariantSet` schema validation, `RefGenome` extraction, the metadata loaders, and per-scorer parity against recorded reference correlations (`tests/test_chrombpnet_*`).
</content>
