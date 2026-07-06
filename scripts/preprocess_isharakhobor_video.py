import os
import csv
import sys
import numpy as np
import argparse

sys.path.append('./')

from utils.helpers import (
    build_isharakhobor_video_index,
    count_video_frames,
    resolve_isharakhobor_video,
)


SPLIT_MAP = {
    'train': 'train',
    'val': 'dev',
    'dev': 'dev',
    'test': 'test',
}


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess isharakhobor dataset from source videos "
            "(video-direct pipeline; no extracted frames required)"
        )
    )
    parser.add_argument(
        '--csv_path', required=True,
        help='Path to sentence mapping CSV '
             '(headers: video-id,sentence-id,start,end,split,sentence)',
    )
    parser.add_argument(
        '--dataset_root', required=True,
        help='Root directory of isharakhobor dataset (contains CLIPS/)',
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='Output directory for annotation npy files',
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    clips_dir = os.path.join(args.dataset_root, 'CLIPS')
    if not os.path.isdir(clips_dir):
        raise FileNotFoundError(f"CLIPS directory not found: {clips_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    video_index = build_isharakhobor_video_index(clips_dir)
    print(f"Indexed {len(video_index)} videos under {clips_dir}")

    rows = []
    with open(args.csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} sentences from CSV")

    splits = {'train': [], 'dev': [], 'test': []}
    missing = 0
    unknown_split = 0

    for row in rows:
        video_id = row['video-id'].strip()
        sentence_id = row['sentence-id'].strip()
        sentence = row['sentence'].strip()
        start_time = row['start'].strip()
        end_time = row['end'].strip()
        split_raw = row['split'].strip().lower()

        split_name = SPLIT_MAP.get(split_raw)
        if split_name is None:
            print(f"Warning: Unknown split '{split_raw}' for {sentence_id}, skipping")
            unknown_split += 1
            continue

        rel_path, use_time_bounds = resolve_isharakhobor_video(
            video_index, video_id, sentence_id
        )
        if rel_path is None:
            missing += 1
            continue

        video_path = os.path.join(clips_dir, rel_path)
        n_frames = count_video_frames(
            video_path,
            start_time=start_time,
            end_time=end_time,
            use_time_bounds=use_time_bounds,
        )
        if n_frames == 0:
            print(f"Warning: No frames found in {video_path}, skipping")
            missing += 1
            continue

        splits[split_name].append({
            'fileid': sentence_id,
            'folder': rel_path,
            'source_type': 'video',
            'text': sentence,
            'gloss': '',
            'sentence_id': sentence_id,
            'video_name': video_id,
            'start_time': start_time,
            'end_time': end_time,
            'use_time_bounds': use_time_bounds,
            'num_frames': n_frames,
        })

    if missing > 0:
        print(f"Warning: {missing} entries had no matching video and were skipped")
    if unknown_split > 0:
        print(f"Warning: {unknown_split} entries had an unrecognized split value")

    total = sum(len(s) for s in splits.values())
    print(f"Total valid samples: {total}")

    for split_name, samples in splits.items():
        data = {i: sample for i, sample in enumerate(samples)}

        np.save(os.path.join(args.output_dir, f'{split_name}_info.npy'), data)
        np.save(os.path.join(args.output_dir, f'{split_name}_info_ml.npy'), data)

        n_sents = len(set(s['sentence_id'] for s in samples))
        print(f"  {split_name}: {len(samples)} samples ({n_sents} unique sentences)")

    print(f"\nAnnotation files saved to {args.output_dir}")
    print("Use --video_root <dataset_root>/CLIPS when running feature extraction.")


if __name__ == '__main__':
    main()
