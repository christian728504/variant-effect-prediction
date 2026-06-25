"""FoldedModelWeights — 5-fold weight container for BPNet-like scorers.

Each fold is a single torch state-dict file (`<MODEL_NAME>/fold_{i}.torch`).
The ChromBPNet/Cherimoya scorers deserialize these themselves; this container
only locates and validates the per-fold files. ChromBPNet h5/tar archives are
converted to this layout once via `scripts/convert_chrombpnet_to_torch.py`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


N_FOLDS = 5


@dataclass(frozen=True)
class FoldedModelWeights:
    """Holds the 5 per-fold torch state-dict file paths for an ensemble."""

    folds: tuple[Path, Path, Path, Path, Path]

    def __post_init__(self) -> None:
        if len(self.folds) != N_FOLDS:
            raise ValueError(f"Exactly {N_FOLDS} folds required, got {len(self.folds)}")

    def __iter__(self) -> Iterator[Path]:
        return iter(self.folds)

    def __getitem__(self, idx: int) -> Path:
        return self.folds[idx]

    def __len__(self) -> int:
        return N_FOLDS

    def __repr__(self) -> str:
        """Multi-line listing of each fold's weight file."""
        lines = [f"FoldedModelWeights(n_folds={len(self.folds)}):"]
        lines.extend(f"  fold {i}: {p}" for i, p in enumerate(self.folds))
        return "\n".join(lines)

    @classmethod
    def from_dir(
        cls,
        dir_path: str | Path,
        pattern: str = "fold_{i}.torch",
    ) -> "FoldedModelWeights":
        """Build from a directory of 5 per-fold torch state-dict files.

        Expects the `<MODEL_NAME>/fold_{i}.torch` layout. `pattern` is
        configurable for legacy naming (e.g. `fold_{i}.final.torch`).
        Validates that each fold file exists.
        """
        dir_path = Path(dir_path)
        paths = []
        for i in range(N_FOLDS):
            p = dir_path / pattern.format(i=i)
            if not p.exists():
                raise FileNotFoundError(f"fold {i} weight not found: {p}")
            paths.append(p)
        return cls(tuple(paths))  # type: ignore[arg-type]
