"""FoldedModelWeights — 5-fold weight container for BPNet-like scorers.

The fold values are intentionally untyped (`Any`) — each model has its own
loader convention. For ChromBPNet the values are `(BytesIO, BytesIO)` tuples
(bias + accessibility); for Cherimoya they are `Path` objects to .torch files.
"""

from __future__ import annotations

import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any


N_FOLDS = 5


@dataclass(frozen=True)
class FoldedModelWeights:
    """Holds 5 per-fold weight handles. The values' type is loader-dependent."""

    folds: tuple[Any, Any, Any, Any, Any]

    def __post_init__(self) -> None:
        if len(self.folds) != N_FOLDS:
            raise ValueError(f"Exactly {N_FOLDS} folds required, got {len(self.folds)}")

    def __iter__(self) -> Iterator[Any]:
        return iter(self.folds)

    def __getitem__(self, idx: int) -> Any:
        return self.folds[idx]

    def __len__(self) -> int:
        return N_FOLDS

    def __repr__(self) -> str:
        """Multi-line listing of each fold's weight source(s).

        Renders Path/str folds as their path and ChromBPNet `(bias, acc)` BytesIO
        blobs as `<BytesIO N bytes>`, so the Stage 4 state report can show exactly
        which weights a scorer was built from.
        """

        def _fmt(v: Any) -> str:
            if isinstance(v, BytesIO):
                return f"<BytesIO {v.getbuffer().nbytes} bytes>"
            return str(v)

        lines = [f"FoldedModelWeights(n_folds={len(self.folds)}):"]
        for i, fold in enumerate(self.folds):
            if isinstance(fold, tuple) and len(fold) == 2:
                bias, acc = fold
                lines.append(f"  fold {i}: bias={_fmt(bias)} acc={_fmt(acc)}")
            else:
                lines.append(f"  fold {i}: {_fmt(fold)}")
        return "\n".join(lines)

    @classmethod
    def from_chrombpnet_tar(
        cls,
        tar_path: str | Path,
        eid: str,
    ) -> "FoldedModelWeights":
        """Extract the 5 (bias, accessibility) BytesIO pairs from a ChromBPNet
        archive. `eid` is the ExpID used in the embedded h5 filenames
        (e.g. `ENCSR000EMT` for GM12878 DNase). The archive layout is:

            ./fold_{N}/model.bias_scaled.fold_{N}.{eid}.h5
            ./fold_{N}/model.chrombpnet_nobias.fold_{N}.{eid}.h5
        """
        tar_path = Path(tar_path)
        folds: list[tuple[BytesIO, BytesIO]] = []
        with tarfile.open(tar_path, "r:*") as tf:
            for fold in range(N_FOLDS):
                bias_name = f"./fold_{fold}/model.bias_scaled.fold_{fold}.{eid}.h5"
                acc_name = f"./fold_{fold}/model.chrombpnet_nobias.fold_{fold}.{eid}.h5"

                bias_member = tf.extractfile(bias_name)
                if bias_member is None:
                    raise FileNotFoundError(f"missing {bias_name!r} in {tar_path}")
                bias_blob = BytesIO(bias_member.read())

                acc_member = tf.extractfile(acc_name)
                if acc_member is None:
                    raise FileNotFoundError(f"missing {acc_name!r} in {tar_path}")
                acc_blob = BytesIO(acc_member.read())

                folds.append((bias_blob, acc_blob))

        return cls(tuple(folds))  # type: ignore[arg-type]

    @classmethod
    def from_cherimoya_dir(
        cls,
        dir_path: str | Path,
        pattern: str = "fold_{i}.final.torch",
    ) -> "FoldedModelWeights":
        """Build from a directory of Cherimoya .torch files.

        Default pattern `fold_{i}.final.torch` matches the user-confirmed
        canonical Cherimoya weight naming.
        """
        dir_path = Path(dir_path)
        paths = []
        for i in range(N_FOLDS):
            p = dir_path / pattern.format(i=i)
            if not p.exists():
                raise FileNotFoundError(f"fold {i} weight not found: {p}")
            paths.append(p)
        return cls(tuple(paths))  # type: ignore[arg-type]

    @classmethod
    def from_chrombpnet_h5_folds(
        cls,
        models_dir: str | Path,
        bias_name: str = "bias_model_scaled.h5",
        acc_name: str = "chrombpnet_nobias.h5",
        fold_subdir: str = "fold_{i}",
    ) -> "FoldedModelWeights":
        """Build from a ChromBPNet `models/fold_N/<bias>.h5 + <acc>.h5` layout.

        This is the primary-cell layout used in the syn59449898 archive
        (e.g. `Microglia_scATAC_ChromBPNet/models/fold_0/bias_model_scaled.h5`),
        distinct from the ENCODE tar layout handled by `from_chrombpnet_tar`.
        Returns (bias_path, acc_path) file-path pairs per fold; ChromBPNet's
        `from_chrombpnet` accepts filenames directly.
        """
        models_dir = Path(models_dir)
        folds: list[tuple[Path, Path]] = []
        for i in range(N_FOLDS):
            fold_dir = models_dir / fold_subdir.format(i=i)
            bias_p = fold_dir / bias_name
            acc_p = fold_dir / acc_name
            for p in (bias_p, acc_p):
                if not p.exists():
                    raise FileNotFoundError(f"fold {i} weight not found: {p}")
            folds.append((bias_p, acc_p))
        return cls(tuple(folds))  # type: ignore[arg-type]
