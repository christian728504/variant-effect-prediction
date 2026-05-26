"""Tangermeme wrapper for AlphaGenome.

AlphaGenome:
    - Input: (B, L, 4) float one-hot (channels-last). Default L = 131_072 bp per
      the alphagenome-pytorch README; the model supports variable length up to 1M.
    - Output: model.predict(...) returns a dict {head: {resolution_bp: tensor}}
      where the tensor has shape (B, L // bin_size, n_tracks_padded). Track dim
      is zero-padded — we index the user-provided track indices directly (caller
      is responsible for staying within the real-track count from the metadata
      module).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from variant_effect_prediction.wrappers.base import ManyTracksWrapperBase


ALPHAGENOME_DEFAULT_SEQ_LEN = 131_072


class AlphaGenomeSummedTrack(ManyTracksWrapperBase):
    """Wraps AlphaGenome's predict() over a single head + resolution.

    `output_name` is the head key (lower-case, e.g. 'atac', 'dnase'). `bin_size`
    is the requested resolution in bp (1 or 128 for ATAC/DNase). `context_len`
    is the input length in bp; n_output_bins is derived as context_len // bin_size.
    """

    def __init__(
        self,
        alphagenome: nn.Module,
        track_idx: Sequence[int],
        eval_window_len: int = 1000,
        output_name: str = "atac",
        bin_size: int = 128,
        context_len: int = ALPHAGENOME_DEFAULT_SEQ_LEN,
        organism_index: int = 0,
        epsilon: float = 1e-8,
    ) -> None:
        if context_len % bin_size != 0:
            raise ValueError(
                f"context_len ({context_len}) must be a multiple of bin_size ({bin_size})"
            )
        # Set instance attributes before super().__init__ (base reads them as class attrs).
        self.bin_size = bin_size
        self.n_output_bins = context_len // bin_size
        super().__init__(alphagenome, track_idx, eval_window_len, epsilon=epsilon)
        self.output_name = output_name
        self.context_len = context_len
        self.organism_index = organism_index

    def _predict_central_tracks(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[-1] != self.context_len:
            raise ValueError(
                f"AlphaGenome expects sequences of length {self.context_len}, "
                f"got {X.shape[-1]}"
            )
        x = X.permute(0, 2, 1).contiguous()  # (B, L, 4)
        out = self.model.predict(
            x,
            organism_index=self.organism_index,
            resolutions=(self.bin_size,),
            heads=(self.output_name,),
        )
        head = out[self.output_name][self.bin_size]  # (B, n_bins, n_tracks_padded)
        head = head[:, self.bin_lo : self.bin_hi, :]
        return head.index_select(-1, self.track_idx)
