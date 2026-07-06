import os
import csv
import numpy as np
import argparse


# CSV 'split' values -> annotation split names
SPLIT_MAP = {
    'train': 'train',
    'val': 'dev',
    'dev': 'dev',
    'test': 'test',
}


def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess isharakhobor dataset: CSV -> annotation npy files"
    )
    parser.add_argument(
        '--csv_path', required=True,
        help='Path to sentence mapping CSV '
             '(headers: video-id,sentence-id,start,end,split,sentence)'
    )
    parser.add_argument(
        '--frame_root', required=True,
        help='Root directory containing extracted frames '
             '(output of extract_frames_isharakhobor.py)'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='Output directory for annotation npy files'
    )
    parser.add_argument(
        '--format', choices=['png', 'webp'], default='png',
        help='Image format of extracted frames (default: png)'
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ext = f'.{args.format}'

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

        # Frames are stored in a folder named after the clip's video-id
        frames_path = os.path.join(args.frame_root, sentence_id)
        if not os.path.isdir(frames_path):
            missing += 1
            continue

        n_frames = len([
            f for f in os.listdir(frames_path)
            if f.endswith(ext)
        ])

        if n_frames == 0:
            print(f"Warning: No frames found in {frames_path}, skipping")
            missing += 1
            continue

        splits[split_name].append({
            'fileid': sentence_id,
            'folder': os.path.join(sentence_id, f'*.{args.format}'),
            'text': sentence,
            'gloss': '',
            'sentence_id': sentence_id,
            'video_name': video_id,
            'start_time': start_time,
            'end_time': end_time,
            'num_frames': n_frames,
        })

    if missing > 0:
        print(f"Warning: {missing} entries had no extracted frames and were skipped")
    if unknown_split > 0:
        print(f"Warning: {unknown_split} entries had an unrecognized split value")

    total = sum(len(s) for s in splits.values())
    print(f"Total valid samples: {total}")

    for split_name, samples in splits.items():
        data = {}
        for i, sample in enumerate(samples):
            data[i] = sample

        np.save(os.path.join(args.output_dir, f'{split_name}_info.npy'), data)
        np.save(os.path.join(args.output_dir, f'{split_name}_info_ml.npy'), data)

        n_sents = len(set(s['sentence_id'] for s in samples))
        print(f"  {split_name}: {len(samples)} samples ({n_sents} unique sentences)")

    print(f"\nAnnotation files saved to {args.output_dir}")


if __name__ == '__main__':
    main()
