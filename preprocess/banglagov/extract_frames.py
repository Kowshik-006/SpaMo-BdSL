"""
Extract frames from Banglagov videos into the layout expected by ViT/MAE feature scripts.

Expects:
  - video_root/
      <Sentence ID>/
        video0.mp4, video1.mp4, ... (5 videos, any names; they are sorted by name)
  - preprocess/Banglagov/{train,dev,test}_info.npy already built (run build_anno_from_excel.py first)

Output:
  - frames_root/features/fullFrame-256x256px/banglagov/{train,dev,test}/<fileid>/
      00001.png, 00002.png, ...

Usage:
  python preprocess/banglagov/extract_frames.py \
    --anno_dir ./preprocess/Banglagov \
    --video_root /path/to/banglagov/videos \
    --frames_root /path/to/banglagov/frames \
    --video_ext .mp4
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None


def extract_frames_from_video(video_path, out_dir, pattern="%05d.png"):
    """Extract every frame as PNG. out_dir is created if needed."""
    if cv2 is None:
        raise ImportError("opencv-python is required: pip install opencv-python")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        name = pattern % idx
        path = out_dir / name
        cv2.imwrite(str(path), frame)
        idx += 1
    cap.release()
    return idx - 1


def main():
    parser = argparse.ArgumentParser(description="Extract Banglagov video frames for feature extraction")
    parser.add_argument("--anno_dir", default="./preprocess/Banglagov", help="Directory with train/dev/test_info.npy")
    parser.add_argument("--video_root", required=True, help="Root with 1922 sentence ID dirs, each with 5 videos")
    parser.add_argument("--frames_root", required=True, help="Output root for frames (e.g. .../banglagov_frames)")
    parser.add_argument("--video_ext", default=".mp4", help="Video extension to look for")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = parser.parse_args()

    video_root = Path(args.video_root)
    frames_root = Path(args.frames_root)
    anno_dir = Path(args.anno_dir)
    # Output: frames_root/features/fullFrame-256x256px/banglagov/<split>/<fileid>/*.png
    base_out = frames_root / "features" / "fullFrame-256x256px" / "banglagov"

    video_ext_lower = args.video_ext.lower()

    for split in args.splits:
        npy_path = anno_dir / f"{split}_info.npy"
        if not npy_path.exists():
            print(f"Skipping {split}: {npy_path} not found. Run build_anno_from_excel.py first.")
            continue
        data = np.load(npy_path, allow_pickle=True).item()
        n = len(data) - 1 if "prefix" in data else len(data)
        indices = [i for i in data if isinstance(i, int)]
        for i in tqdm(indices, desc=f"Extract frames [{split}]"):
            rec = data[i]
            fileid = rec["fileid"]
            # fileid = "<sentence_id>_signer_<0..4>"
            parts = fileid.rsplit("_signer_", 1)
            if len(parts) != 2:
                sentence_id = fileid
                signer_idx = 0
            else:
                sentence_id, signer_idx = parts[0], int(parts[1])
            sentence_dir = video_root / sentence_id
            if not sentence_dir.is_dir():
                print(f"Warning: missing directory {sentence_dir}, skipping {fileid}")
                continue
            # List video files and sort for stable order
            video_files = sorted(
                [f for f in sentence_dir.iterdir() if f.suffix.lower() == video_ext_lower or f.suffix.lower() == ".avi"]
            )
            if len(video_files) <= signer_idx:
                print(f"Warning: expected at least {signer_idx + 1} videos in {sentence_dir}, skipping {fileid}")
                continue
            video_path = video_files[signer_idx]
            out_dir = base_out / split / fileid
            n_frames = extract_frames_from_video(video_path, out_dir)
            if n_frames == 0:
                print(f"Warning: no frames extracted from {video_path}")

    print(f"Frames written under {base_out}. Use --video_root {frames_root} for feature extraction.")


if __name__ == "__main__":
    main()