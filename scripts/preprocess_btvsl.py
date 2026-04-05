import os
import csv
import random
import numpy as np
import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess BTVSL dataset: CSV -> annotation npy files"
    )
    parser.add_argument(
        '--csv_path', required=True,
        help='Path to sentences.csv'
    )
    parser.add_argument(
        '--frame_root', required=True,
        help='Root directory containing extracted frames (output of extract_frames_btvsl.py)'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='Output directory for annotation npy files'
    )
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--dev_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    assert abs(args.train_ratio + args.dev_ratio + args.test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    with open(args.csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} sentences from CSV")

    all_samples = []
    missing = 0

    for row in rows:
        sentence_id = row['sentence_id'].strip()
        video_name = row['video_name'].strip()
        sentence = row['sentence'].strip()
        start_time = row['start_time'].strip()
        end_time = row['end_time'].strip()

        frames_path = os.path.join(args.frame_root, sentence_id)
        if not os.path.isdir(frames_path):
            missing += 1
            continue

        n_frames = len([
            f for f in os.listdir(frames_path)
            if f.endswith('.png')
        ])

        if n_frames == 0:
            print(f"Warning: No frames found in {frames_path}, skipping")
            missing += 1
            continue

        all_samples.append({
            'fileid': sentence_id,
            'folder': os.path.join(sentence_id, '*.png'),
            'text': sentence,
            'gloss': '',
            'sentence_id': sentence_id,
            'video_name': video_name,
            'start_time': start_time,
            'end_time': end_time,
            'num_frames': n_frames,
        })

    if missing > 0:
        print(f"Warning: {missing} entries had no extracted frames and were skipped")

    print(f"Total valid samples: {len(all_samples)}")

    unique_video_names = sorted(set(s['video_name'] for s in all_samples))
    print(f"Unique source videos: {len(unique_video_names)}")

    random.seed(args.seed)
    random.shuffle(unique_video_names)

    n_train = int(len(unique_video_names) * args.train_ratio)
    n_dev = int(len(unique_video_names) * args.dev_ratio)

    train_vids = set(unique_video_names[:n_train])
    dev_vids = set(unique_video_names[n_train:n_train + n_dev])
    test_vids = set(unique_video_names[n_train + n_dev:])

    splits = {
        'train': [s for s in all_samples if s['video_name'] in train_vids],
        'dev': [s for s in all_samples if s['video_name'] in dev_vids],
        'test': [s for s in all_samples if s['video_name'] in test_vids],
    }

    for split_name, samples in splits.items():
        data = {}
        for i, sample in enumerate(samples):
            data[i] = sample

        np.save(os.path.join(args.output_dir, f'{split_name}_info.npy'), data)
        np.save(os.path.join(args.output_dir, f'{split_name}_info_ml.npy'), data)

        n_vids = len(set(s['video_name'] for s in samples))
        print(f"  {split_name}: {len(samples)} samples ({n_vids} source videos)")

    print(f"\nAnnotation files saved to {args.output_dir}")


if __name__ == '__main__':
    main()
