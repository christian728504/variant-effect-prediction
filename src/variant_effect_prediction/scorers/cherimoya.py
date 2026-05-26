"""CherimoyaVariantScorer — 5-fold ensemble; forward-only; uses tangermeme.predict."""

from __future__ import annotations

import torch
from cherimoya import Cherimoya
from tangermeme.predict import predict

from variant_effect_prediction.references import RefGenome
from variant_effect_prediction.scorers.base import VariantScorer
from variant_effect_prediction.scoring import _squeeze_profile, one_hot_batch
from variant_effect_prediction.weights import FoldedModelWeights


class CherimoyaVariantScorer(VariantScorer):
    name = "cherimoya"
    context_len = 2114

    def __init__(
        self,
        *,
        folded_weights: FoldedModelWeights,
        ref_genome: RefGenome,
        eval_window_len: int = 1000,
        batch_size: int = 64,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        cherimoya_compile: bool = False,
    ) -> None:
        if eval_window_len != 1000:
            raise ValueError(
                "Cherimoya inherits ChromBPNet's 1 kb count window; "
                "eval_window_len must be 1000"
            )
        super().__init__(
            ref_genome=ref_genome,
            eval_window_len=eval_window_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        self.folded_weights = folded_weights
        self.cherimoya_compile = cherimoya_compile

    def _load_fold(self, fold: int) -> torch.nn.Module:
        weight_path = str(self.folded_weights[fold])
        return Cherimoya.load(
            weight_path, device=self.device, compile=self.cherimoya_compile
        )

    def _predict_alleles(self, allele1_seqs, allele2_seqs):
        X1 = one_hot_batch(allele1_seqs)
        X2 = one_hot_batch(allele2_seqs)

        a1_counts, a2_counts = [], []
        a1_profs, a2_profs = [], []
        for fold in range(5):
            model = self._load_fold(fold)
            model.eval()
            with torch.no_grad():
                a1_prof, a1_lc = predict(
                    model, X1, batch_size=self.batch_size,
                    device=self.device, dtype=self.dtype,
                )
                a2_prof, a2_lc = predict(
                    model, X2, batch_size=self.batch_size,
                    device=self.device, dtype=self.dtype,
                )
            a1_counts.append(a1_lc.cpu())             # (N, 1)
            a2_counts.append(a2_lc.cpu())
            a1_profs.append(_squeeze_profile(a1_prof.cpu()))   # (N, W)
            a2_profs.append(_squeeze_profile(a2_prof.cpu()))
            del model
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        allele1_counts = torch.stack(a1_counts, dim=1)   # (N, 5, 1)
        allele2_counts = torch.stack(a2_counts, dim=1)
        allele1_profile = torch.stack(a1_profs, dim=1)   # (N, 5, W)
        allele2_profile = torch.stack(a2_profs, dim=1)
        return allele1_counts, allele2_counts, allele1_profile, allele2_profile
