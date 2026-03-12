"""
Create a smaller Banglagov subset by sub-sampling entries from the existing
`preprocess/Banglagov/{train,dev,test}_info.npy` and `_info_ml.npy` files.

You can then point the feature extraction scripts and training config to
this subset anno_root so that the whole pipeline runs only on a fraction
of the dataset.

Example:
  python preprocess/banglagov/make_subset.py \\
    --anno_dir ./preprocess/Banglagov \\
    --out_dir ./preprocess/Banglagov_subset \\
    --fraction 0.1

Then run feature extraction with:
  --anno_root ./preprocess/Banglagov_subset
  --video_root /home/mdsr/CSLT-Kowshik-Adiba/banglagov/frame_root
"""

import argparse
from pathlib import Path
import numpy as np


def build_subset(
    anno_dir: Path,
    out_dir: Path,
    fraction: float,
    max_per_split: int | None,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev", "test"):
        src_info = anno_dir / f"{split}_info.npy"
        src_info_ml = anno_dir / f"{split}_info_ml.npy"
        if not src_info.exists() or not src_info_ml.exists():
            print(f"Skipping {split}: {src_info} or {src_info_ml} not found.")
            continue

        data = np.load(src_info, allow_pickle=True).item()
        data_ml = np.load(src_info_ml, allow_pickle=True).item()

        # Integer keys 0..N-1 are samples; 'prefix' is metadata.
        idx_keys = sorted(k for k in data.keys() if isinstance(k, int))
        n = len(idx_keys)
        if n == 0:
            print(f"No samples found in {src_info}, skipping.")
            continue

        target_n = int(round(n * fraction))
        if target_n < 1:
            target_n = 1
        if max_per_split is not None:
            target_n = min(target_n, max_per_split)

        # Random subset of indices
        chosen = sorted(rng.choice(idx_keys, size=target_n, replace=False))

        subset = {"prefix": data.get("prefix", "banglagov")}
        subset_ml = {"prefix": data_ml.get("prefix", "banglagov")}
        for new_i, old_i in enumerate(chosen):
            subset[new_i] = data[old_i]
            subset_ml[new_i] = data_ml[old_i]

        dst_info = out_dir / f"{split}_info.npy"
        dst_info_ml = out_dir / f"{split}_info_ml.npy"
        np.save(dst_info, subset)
        np.save(dst_info_ml, subset_ml)
        print(
            f"{split}: kept {len(chosen)}/{n} samples "
            f"-> {dst_info}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a smaller Banglagov subset by sub-sampling annotation npy files."
    )
    parser.add_argument(
        "--anno_dir",
        default="./preprocess/Banglagov",
        help="Directory containing {train,dev,test}_info.npy and _info_ml.npy.",
    )
    parser.add_argument(
        "--out_dir",
        default="./preprocess/Banglagov_subset",
        help="Where to write the subset annotation files.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Fraction of samples to keep per split (0 < fraction ≤ 1).",
    )
    parser.add_argument(
        "--max_per_split",
        type=int,
        default=None,
        help="Optional hard cap on number of samples per split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sub-sampling.",
    )
    args = parser.parse_args()

    if not (0 < args.fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1].")

    anno_dir = Path(args.anno_dir)
    out_dir = Path(args.out_dir)
    build_subset(
        anno_dir=anno_dir,
        out_dir=out_dir,
        fraction=args.fraction,
        max_per_split=args.max_per_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

