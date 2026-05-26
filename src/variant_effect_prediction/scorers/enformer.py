"""EnformerVariantScorer."""

from __future__ import annotations

from variant_effect_prediction.scorers._many_tracks import _ManyTracksVariantScorer
from variant_effect_prediction.wrappers.enformer import (
    EnformerSummedTrack,
    ENFORMER_SEQ_LEN,
)


class EnformerVariantScorer(_ManyTracksVariantScorer):
    name = "enformer"
    context_len = ENFORMER_SEQ_LEN
    wrapper_cls = EnformerSummedTrack
