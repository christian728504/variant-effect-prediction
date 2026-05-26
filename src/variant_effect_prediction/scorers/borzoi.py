"""BorzoiVariantScorer."""

from __future__ import annotations

from variant_effect_prediction.scorers._many_tracks import _ManyTracksVariantScorer
from variant_effect_prediction.wrappers.borzoi import (
    BorzoiSummedTrack,
    BORZOI_SEQ_LEN,
)


class BorzoiVariantScorer(_ManyTracksVariantScorer):
    name = "borzoi"
    context_len = BORZOI_SEQ_LEN
    wrapper_cls = BorzoiSummedTrack
