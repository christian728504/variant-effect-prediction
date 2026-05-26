# Descriptions

## Files

src/variant_effect_prediction/metadata/data/
├── alphagenome-supplemental-table-2.parquet
├── alphagenome-track-metadata.parquet
├── borzoi-track-metadata.parquet
├── enformer-targets-human.parquet
└── README.md

## `alphagenome-supplemental-table-2.parquet`

From the supplementary information of the [Alphagenome paper](https://doi.org/10.1038/s41586-0). Specifically, Table 2 from the [supplementary tables](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-025-10014-0/MediaObjects/41586_2025_10014_MOESM3_ESM.xlsx).

## `alphagenome-track-metadata.parquet`

From the [genomicsxai/alphagenome-pytorch repo](https://github.com/genomicsxai/alphagenome-pytorch.git). Specifically under `src/alphagenome-pytorch/data/track_metadata_human.parquet`. Download [here](https://github.com/genomicsxai/alphagenome-pytorch/raw/refs/heads/main/src/alphagenome_pytorch/data/track_metadata_human.parquet). This file can be reproduce by running the following script from their repo: `scripts/extract_track_metadata.py`.

## `borzoi-track-metadata.parquet`

From the [calico/borzoi repo](https://github.com/calico/borzoi.git). Specifically at the beginning of the `README.md`. Download [here](https://raw.githubusercontent.com/calico/borzoi/main/examples/targets_human.txt)

## `enformer-targets-human.parquet`

From the [google-deepmind/deepmind-research repo](https://github.com/google-deepmind/deepmind-research.git). Specifically, `enformer` project folder's `README.md`, which describes that dataset, and internally links to the `manuscripts/cross2020` folder, which contains the file. Download [here](https://raw.githubusercontent.com/calico/basenji/refs/heads/master/manuscripts/cross2020/targets_human.txt).

## File Format Conversion

All of the aforementioned files were converted to parquet files for space savings and format standardization.
