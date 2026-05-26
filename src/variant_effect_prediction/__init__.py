"""Variant effect prediction scoring API.

Public surface:
    VariantSet, VariantSetSchemaError    — variant table with canonical schema
    RefGenome                             — pysam-backed reference genome cache
    FoldedModelWeights                    — 5-fold weight container (BPNet-like)
    VariantScorer                         — scorer ABC
    EnformerVariantScorer
    BorzoiVariantScorer
    AlphaGenomeVariantScorer
    ChromBPNetVariantScorer
    CherimoyaVariantScorer
"""

from variant_effect_prediction.variantset import VariantSet, VariantSetSchemaError
from variant_effect_prediction.references import RefGenome
from variant_effect_prediction.weights import FoldedModelWeights
from variant_effect_prediction.scorers.base import VariantScorer

__all__ = [
    "VariantSet",
    "VariantSetSchemaError",
    "RefGenome",
    "FoldedModelWeights",
    "VariantScorer",
]
