"""Tangermeme-compatible nn.Module wrappers for the many-tracks models."""

from variant_effect_prediction.wrappers.base import ManyTracksWrapperBase
from variant_effect_prediction.wrappers.enformer import EnformerSummedTrack
from variant_effect_prediction.wrappers.borzoi import BorzoiSummedTrack
from variant_effect_prediction.wrappers.alphagenome import AlphaGenomeSummedTrack

__all__ = [
    "ManyTracksWrapperBase",
    "EnformerSummedTrack",
    "BorzoiSummedTrack",
    "AlphaGenomeSummedTrack",
]
