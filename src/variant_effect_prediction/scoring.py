"""Scoring primitives shared across scorers.

`compute_logfc` matches the variant-scorer reference implementation
(variant_scoring.py:217-233) but operates in log space throughout to avoid
overflow on long sequences:

    logFC = ( mean_allele2_log - mean_allele1_log ) / ln(2)
          = log2( exp(mean_allele2) / exp(mean_allele1) )

where the mean is taken across the leading "ensemble" dim (folds for BPNet-like
models, selected tracks for many-tracks models). allele1/allele2 follow the
variant-scorer naming (allele1 is the QTL reference-like allele).
"""

from __future__ import annotations

import math
import string

import torch
from tangermeme.utils import one_hot_encode

KEEP = "ACGT"
IUPAC_CODES = "ACGTURYSWKMBDHVN.-"
IGNORE = [c for c in IUPAC_CODES if c not in KEEP]


def one_hot_batch(seqs: list[str]) -> torch.Tensor:
    """Return a (B, 4, L) float32 one-hot tensor over A/C/G/T (N → 0 column)."""
    encoded = [one_hot_encode(s, alphabet=KEEP, ignore=IGNORE).float() for s in seqs]
    return torch.stack(encoded)


def compute_logfc(
    allele1_pred: torch.Tensor, allele2_pred: torch.Tensor
) -> torch.Tensor:
    """logFC over a batch of (N, ensemble_dim, last_dim) log-counts predictions.

    Inputs are in log space (natural log for ChromBPNet's count head; log1p(sum)
    for the many-tracks wrappers). Returns a (N,) base-2 logFC tensor (float64),
    log2(allele2 / allele1).
    """
    if allele1_pred.shape != allele2_pred.shape:
        raise ValueError(
            f"allele1_pred and allele2_pred shape mismatch: "
            f"{tuple(allele1_pred.shape)} vs {tuple(allele2_pred.shape)}"
        )
    if allele1_pred.ndim != 3:
        raise ValueError(
            f"expected (N, dim, last_dim); got shape {tuple(allele1_pred.shape)}"
        )

    a1_mean = allele1_pred.to(torch.float64).mean(dim=1).squeeze(-1)
    a2_mean = allele2_pred.to(torch.float64).mean(dim=1).squeeze(-1)
    return (a2_mean - a1_mean) / math.log(2.0)


def _squeeze_profile(prof: torch.Tensor) -> torch.Tensor:
    """Reduce a model profile to (N, W): (N, 1, W) → (N, W); (N, W) → (N, W)."""
    if prof.ndim == 3 and prof.shape[1] == 1:
        return prof[:, 0, :]
    if prof.ndim == 2:
        return prof
    return prof.reshape(prof.shape[0], -1)


def _softmax_np(x):
    """Profile softmax matching variant_effect/utils.py (mean-subtracted, temp=1)."""
    import numpy as np

    norm = x - np.mean(x, axis=1, keepdims=True)
    e = np.exp(norm)
    return e / np.sum(e, axis=1, keepdims=True)


def compute_jsd(
    allele1_profile: torch.Tensor,
    allele2_profile: torch.Tensor,
    logfc,
    profile_normalization: str = "softmax",
):
    """Signed Jensen-Shannon distance between the two alleles' profiles.

    Matches the variant_effect reference (`scoring.py`): the per-fold raw profiles
    are fold-averaged (mean over the ensemble dim, in logit/count space), normalized
    to probabilities, and compared with `scipy.spatial.distance.jensenshannon`
    (base 2), then signed by logFC.

    Parameters
    ----------
    allele1_profile, allele2_profile : torch.Tensor, shape (N, ensemble, W)
        Raw per-fold profiles (logits for `softmax`, non-negative counts for `sum`).
    logfc : array-like, shape (N,)
        Base-2 logFC, used only for its sign.
    profile_normalization : {"softmax", "sum"}
        "softmax" for logit heads (ChromBPNet / Cherimoya); "sum" for non-negative
        heads (AlphaGenome).

    Returns
    -------
    numpy.ndarray, shape (N,), float64
        Signed JSD in [-1, 1].
    """
    import numpy as np
    from scipy.spatial.distance import jensenshannon

    if allele1_profile.shape != allele2_profile.shape:
        raise ValueError(
            f"profile shape mismatch: {tuple(allele1_profile.shape)} vs "
            f"{tuple(allele2_profile.shape)}"
        )
    if allele1_profile.ndim != 3:
        raise ValueError(
            f"expected (N, ensemble, W); got shape {tuple(allele1_profile.shape)}"
        )

    # Fold-average raw profiles (matches reference: mean over folds, then normalize).
    p1 = allele1_profile.to(torch.float64).mean(dim=1).cpu().numpy()  # (N, W)
    p2 = allele2_profile.to(torch.float64).mean(dim=1).cpu().numpy()

    if profile_normalization == "softmax":
        p1, p2 = _softmax_np(p1), _softmax_np(p2)
    elif profile_normalization == "sum":
        p1 = p1 / np.maximum(p1.sum(axis=1, keepdims=True), 1e-8)
        p2 = p2 / np.maximum(p2.sum(axis=1, keepdims=True), 1e-8)
    else:
        raise ValueError(
            f"Unknown profile_normalization: {profile_normalization!r}. "
            f"Expected 'softmax' or 'sum'."
        )

    jsd = jensenshannon(p1, p2, base=2, axis=1)  # (N,) JS distance in [0, 1]
    sign = np.sign(np.asarray(logfc, dtype=np.float64))
    return (jsd * sign).astype(np.float64)
