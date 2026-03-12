"""
Build train/dev/test_info.npy and _info_ml.npy for Banglagov from the Excel annotation file.

Expected Excel columns:
  - Sentence ID: name of the directory containing 5 videos (same sentence, 5 signers)
  - Natural sentence: natural language translation
  - Sign Sentence: gloss sequence

Dataset layout:
  - video_root/
    - <Sentence ID>/   (1922 directories)
      - video0.mp4 (or .avi, etc.)  # 5 videos per directory
      - video1.mp4
      - ...
      - video4.mp4

Usage:
  python preprocess/banglagov/build_anno_from_excel.py \
    --excel /path/to/annotations.xlsx \
    --video_root /path/to/banglagov/videos \
    --out_dir ./preprocess/Banglagov \
    --train_ratio 0.7 --dev_ratio 0.15
"""

import os
import argparse
import numpy as np
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None

# Fallback: open xlsx with openpyxl if needed
def read_excel(path):
    if pd is not None:
        return pd.read_excel(path)
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            raise ValueError("Excel file is empty")
        header = [str(c).strip() for c in rows[0]]
        data = []
        for r in rows[1:]:
            data.append(dict(zip(header, r)))
        return pd.DataFrame(data)
    except Exception as e:
        raise RuntimeError(f"Could not read Excel file. Install pandas or openpyxl. Error: {e}")


def normalize_text(text):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text).strip()
    if text and not text.endswith('.'):
        text = text + "."
    return text


def main():
    parser = argparse.ArgumentParser(description="Build Banglagov annotation .npy from Excel")
    parser.add_argument("--excel", required=True, help="Path to Excel file (Sentence ID, Natural sentence, Sign Sentence)")
    parser.add_argument("--video_root", required=True, help="Root directory containing 1922 sentence ID directories, each with 5 videos")
    parser.add_argument("--out_dir", default="./preprocess/Banglagov", help="Output directory for train/dev/test_info.npy and _info_ml.npy")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--dev_ratio", type=float, default=0.15)
    parser.add_argument("--video_ext", default=".mp4", help="Video extension, e.g. .mp4, .avi")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    df = read_excel(args.excel)

    print(df.columns)

    # Normalize column names (allow common variants)
    col_map = {}
    for c in df.columns:
        cnorm = str(c).strip().lower()
        if "sentence id" in cnorm or "sentence_id" in cnorm:
            col_map[c] = "sentence_id"
        elif "natural sentence" in cnorm or "natural_sentence" in cnorm:
            col_map[c] = "natural_sentence"
        elif "sign sentence" in cnorm or "sign_sentence" in cnorm or "gloss" in cnorm:
            col_map[c] = "sign_sentence"
    df = df.rename(columns=col_map)
    if "sentence_id" not in df.columns or "natural_sentence" not in df.columns or "sign_sentence" not in df.columns:
        raise ValueError(
            "Excel must have columns: Sentence ID, Natural Sentence, Sign Sentence. "
            f"Found: {list(df.columns)}"
        )

    video_root = Path(args.video_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split sentence IDs into train / dev / test
    sentence_ids = df["sentence_id"].astype(str).unique().tolist()
    n = len(sentence_ids)
    np.random.shuffle(sentence_ids)
    n_train = int(n * args.train_ratio)
    n_dev = int(n * args.dev_ratio)
    n_test = n - n_train - n_dev
    train_ids = set(sentence_ids[:n_train])
    dev_ids = set(sentence_ids[n_train : n_train + n_dev])
    test_ids = set(sentence_ids[n_train + n_dev :])

    id_to_split = {}
    for sid in train_ids:
        id_to_split[sid] = "train"
    for sid in dev_ids:
        id_to_split[sid] = "dev"
    for sid in test_ids:
        id_to_split[sid] = "test"

    # Build one sample per video: 5 videos per sentence
    num_signers = 5
    records = []
    for _, row in df.iterrows():
        sentence_id = str(row["sentence_id"]).strip()
        natural = normalize_text(row["natural_sentence"])
        gloss = str(row["sign_sentence"]).strip() if pd.notna(row["sign_sentence"]) else ""
        split = id_to_split.get(sentence_id, "train")

        for signer_idx in range(num_signers):
            fileid = f"{sentence_id}_signer_{signer_idx}"
            # Path under video_root/features/fullFrame-256x256px/ (see get_img_list)
            folder = f"banglagov/{split}/{fileid}/*.png"

            records.append({
                "fileid": fileid,
                "folder": folder,
                "split": split,
                "signer": f"signer_{signer_idx}",
                "gloss": gloss,
                "text": natural,
                "num_frames": -1,
                "original_info": f"{sentence_id}|{signer_idx}|{natural}|{gloss}",
                "tag": "banglagov",
            })

    # Build dict-style structure like Phoenix14T: keys 0,1,...,N-1 and 'prefix'
    data = {"prefix": "banglagov"}
    for i, r in enumerate(records):
        data[i] = r

    for split in ("train", "dev", "test"):
        split_records = [r for r in records if r["split"] == split]
        split_data = {"prefix": "banglagov"}
        for i, r in enumerate(split_records):
            split_data[i] = r

        np.save(out_dir / f"{split}_info.npy", split_data)
        np.save(out_dir / f"{split}_info_ml.npy", split_data)
        print(f"Saved {split}: {len(split_records)} samples -> {out_dir / f'{split}_info.npy'}")

    print(f"Total samples: {len(records)} (train={n_train*num_signers}, dev={n_dev*num_signers}, test={n_test*num_signers})")
    print("Next step: run frame extraction, then ViT/MAE feature extraction (see README_BANGLAGOV.md).")


if __name__ == "__main__":
    main()
