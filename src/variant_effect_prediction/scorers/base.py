"""VariantScorer ABC and shared archetype bases."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import polars as pl
import torch

from variant_effect_prediction.references import RefGenome
from variant_effect_prediction.scoring import compute_jsd, compute_logfc
from variant_effect_prediction.variantset import VariantSet


class VariantScorer(ABC):
    """Abstract base class for all variant scorers.

    Subclasses set the `name` and `context_len` class attrs and implement
    `_predict_alleles(allele1_seqs, allele2_seqs)` returning
    `(allele1_counts, allele2_counts, allele1_profile, allele2_profile)`:
    the count tensors are (N, ensemble_dim, last_dim) log-counts; the profile
    tensors are (N, ensemble_dim, W) raw profiles, or `None` when the model does
    not expose a profile (then `jsd` is null). `profile_normalization` controls
    how profiles become probabilities for JSD ("softmax" for logit heads).
    """

    name: ClassVar[str]
    context_len: ClassVar[int]
    profile_normalization: ClassVar[str] = "softmax"

    def __init__(
        self,
        *,
        ref_genome: RefGenome,
        eval_window_len: int = 1000,
        batch_size: int = 32,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.ref_genome = ref_genome
        self.eval_window_len = eval_window_len
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

    def __repr__(self) -> str:
        """Comprehensive, multi-line state report (Stage 4 §7).

        Always reports the core knobs (name, windows, batch, device, dtype,
        ref FASTA). The dispatch in `bench_config.build_scorer` additionally sets
        repr-only attrs — `accession`, `assay_type`, `track_indices`,
        `weights_path` — which are shown when present. BPNet-like scorers also
        carry a `folded_weights` whose own repr is nested in.
        """
        cls = type(self).__name__
        lines = [
            f"{cls}(",
            f"  name={getattr(self, 'name', '?')!r}",
            f"  context_len={getattr(self, 'context_len', '?')}",
            f"  eval_window_len={self.eval_window_len}",
            f"  batch_size={self.batch_size}",
            f"  device={self.device!r}",
            f"  dtype={self.dtype}",
            f"  profile_normalization={self.profile_normalization!r}",
            f"  ref_genome={getattr(self.ref_genome, 'fasta_path', self.ref_genome)}",
        ]
        for attr in ("accession", "assay_type", "track_indices", "weights_path"):
            val = getattr(self, attr, None)
            if val is not None:
                lines.append(f"  {attr}={val!r}")
        fw = getattr(self, "folded_weights", None)
        if fw is not None:
            lines.append("  folded_weights=" + repr(fw).replace("\n", "\n  "))
        lines.append(")")
        return "\n".join(lines)

    @abstractmethod
    def _predict_alleles(
        self, allele1_seqs: list[str], allele2_seqs: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return (allele1_counts, allele2_counts, allele1_profile, allele2_profile).

        Counts are (N, ensemble_dim, last_dim) log-counts. Profiles are
        (N, ensemble_dim, W) raw profiles, or None if the model exposes no profile.
        """

    def _make_sequences(self, df: pl.DataFrame) -> tuple[list[str], list[str]]:
        allele1_seqs: list[str] = []
        allele2_seqs: list[str] = []
        for chrom, start, allele1, allele2 in zip(
            df["chrom"].to_list(),
            df["start"].to_list(),
            df["allele1"].to_list(),
            df["allele2"].to_list(),
        ):
            s1, s2 = self.ref_genome.extract_alleles(
                chrom, int(start), allele1, allele2, self.context_len
            )
            allele1_seqs.append(s1)
            allele2_seqs.append(s2)
        return allele1_seqs, allele2_seqs

    def score(self, vs: VariantSet) -> pl.DataFrame:
        """Score the rows of `vs` where isused == True; return a result DataFrame.

        Columns: chrom, start, end, allele1, allele2, effect_size, pvalue, raw,
                 allele1_pred (Array(Float32, (dim, 1))), allele2_pred (same),
                 logfc (Float64) = log2(allele2 / allele1),
                 jsd (Float64) = signed Jensen-Shannon distance (softmax-normalized
                 for BPNet-like logit heads, sum-normalized for many-tracks
                 coverage heads; null only if a scorer exposes no profile).
        """
        used = vs.used().df
        a1_seqs, a2_seqs = self._make_sequences(used)
        a1_pred, a2_pred, a1_prof, a2_prof = self._predict_alleles(a1_seqs, a2_seqs)
        logfc = compute_logfc(a1_pred, a2_pred).cpu().numpy()

        if a1_prof is not None and a2_prof is not None:
            jsd = compute_jsd(
                a1_prof, a2_prof, logfc, profile_normalization=self.profile_normalization
            )
        else:
            jsd = [None] * used.height  # length-N all-null column

        # Each row's prediction is a 2D (dim, last_dim) array — store as a Polars Array.
        a1_rows = [row.tolist() for row in a1_pred.detach().cpu().to(torch.float32)]
        a2_rows = [row.tolist() for row in a2_pred.detach().cpu().to(torch.float32)]
        array_dtype = pl.Array(pl.Float32, (a1_pred.shape[1], a1_pred.shape[2]))

        return pl.DataFrame(
            {
                "chrom": used["chrom"],
                "start": used["start"],
                "end": used["end"],
                "allele1": used["allele1"],
                "allele2": used["allele2"],
                "effect_size": used["effect_size"],
                "pvalue": used["pvalue"],
                # Carry the original row struct so notebooks can recover any
                # dataset-specific column (e.g. obs.label) for metric labels.
                "raw": used["raw"],
                "allele1_pred": pl.Series("allele1_pred", a1_rows, dtype=array_dtype),
                "allele2_pred": pl.Series("allele2_pred", a2_rows, dtype=array_dtype),
                "logfc": pl.Series("logfc", logfc, dtype=pl.Float64),
                "jsd": pl.Series("jsd", jsd, dtype=pl.Float64),
            }
        )
