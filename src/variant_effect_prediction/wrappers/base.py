"""Shared logic for many-tracks model wrappers.

`ManyTracksWrapperBase.forward(X)` expects (B, 4, L) — the tangermeme convention.
Subclasses overrride `_predict_central_tracks(X)` to return (B, n_center_bins,
n_tracks) on the model's native scale (counts, not logits). The base class then
sums across central bins and returns log counts of shape (B, n_tracks).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _center_bin_range(n_output_bins: int, n_center_bins: int) -> tuple[int, int]:
    if n_center_bins > n_output_bins:
        raise ValueError(
            f"requested {n_center_bins} center bins exceeds model output "
            f"({n_output_bins} bins)"
        )
    lo = n_output_bins // 2 - n_center_bins // 2
    return lo, lo + n_center_bins


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class ManyTracksWrapperBase(nn.Module):
    """Adapter base for many-tracks models.

    Subclasses must set `bin_size` (bp/bin) and `n_output_bins` (model's native
    output bin count for the chosen head/resolution) before calling `super().__init__`.
    The center `eval_window_len` bp are converted to a bin range with ceiling
    division: a 1000 bp request against a 128 bp/bin model becomes 8 bins (1024 bp).
    """

    bin_size: int
    n_output_bins: int

    def __init__(
        self,
        model: nn.Module,
        track_idx: Sequence[int],
        eval_window_len: int,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if not hasattr(self, "bin_size") or not hasattr(self, "n_output_bins"):
            raise TypeError(
                "subclass must set class attributes bin_size and n_output_bins "
                "before calling super().__init__()"
            )
        self.model = model
        self.eval_window_len = eval_window_len
        self.epsilon = epsilon
        n_center_bins = max(1, _ceil_div(eval_window_len, self.bin_size))
        self.n_center_bins = n_center_bins
        self.bin_lo, self.bin_hi = _center_bin_range(self.n_output_bins, n_center_bins)
        self.register_buffer(
            "track_idx",
            torch.as_tensor(list(track_idx), dtype=torch.long),
            persistent=False,
        )

    def _predict_central_tracks(self, X: torch.Tensor) -> torch.Tensor:
        """Return (B, n_center_bins, n_tracks) on the model's native scale.

        Subclasses implement this — they run the model, slice to (bin_lo, bin_hi)
        across the bin axis, and gather the requested tracks.
        """
        raise NotImplementedError

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 4, L) → (profile, log_counts), matching the (profile, counts) order
        that `tangermeme.predict` unpacks (same convention as the BPNet models).

        - `log_counts` = log(Σ_bins counts + epsilon) per track, shape (B, n_tracks).
          The true log of summed counts (epsilon guards log(0); matches
          `variant_effect/alpha_bench.py`).
        - `profile` = per-bin counts summed over the selected tracks, shape
          (B, n_center_bins). Non-negative coverage → the scorer sum-normalizes it
          for JSD (`variant_effect` uses sum normalization for coverage heads).
        """
        out = self._predict_central_tracks(X)  # (B, n_center_bins, n_tracks)
        log_counts = torch.log(out.sum(dim=1) + self.epsilon)  # (B, n_tracks)
        profile = out.sum(dim=2)  # (B, n_center_bins)
        return profile, log_counts
