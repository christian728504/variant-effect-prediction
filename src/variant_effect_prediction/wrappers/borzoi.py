"""Tangermeme wrapper for Borzoi.

Borzoi:
    - Input: (B, 4, 524_288) float one-hot.
    - Output: (B, n_tracks, 6144) channels-first (per borzoi_pytorch source).
    - bin_size = 32 bp, central 6144 bins (~196 kb of context).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from variant_effect_prediction.wrappers.base import ManyTracksWrapperBase


BORZOI_SEQ_LEN = 524_288


class BorzoiSummedTrack(ManyTracksWrapperBase):
    bin_size = 32
    n_output_bins = 6144

    def __init__(
        self,
        borzoi: nn.Module,
        track_idx: Sequence[int],
        eval_window_len: int = 1000,
        is_human: bool = True,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(borzoi, track_idx, eval_window_len, epsilon=epsilon)
        self.is_human = is_human

    def _predict_central_tracks(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[-1] != BORZOI_SEQ_LEN:
            raise ValueError(
                f"Borzoi expects sequences of length {BORZOI_SEQ_LEN}, got {X.shape[-1]}"
            )
        out = self.model(X, is_human=self.is_human)  # (B, n_tracks, 6144)
        out = out[:, :, self.bin_lo : self.bin_hi]  # (B, n_tracks, n_center)
        out = out.index_select(1, self.track_idx)  # (B, len(track_idx), n_center)
        return out.permute(0, 2, 1).contiguous()  # (B, n_center, n_tracks)
