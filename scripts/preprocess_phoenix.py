import os
import csv
import numpy as np
import argparse

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
    with open(args.csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f,delimiter='|')
        header = next(reader)

        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} sentences from CSV")

    splits = {'train': [], 'dev': [], 'test': []}
    missing = 0
    unknown_split = 0

    '''
    ['name', 'video', 'start', 'end', 'speaker', 'orth', 'translation', 'split']
    '''

    for row in rows:
        name = row['name'].strip()
        video = row['video'].strip()
        start = row['start'].strip()
        end = row['end'].strip()
        speaker = row['speaker'].strip()
        orth = row['orth'].strip()
        translation = row['translation'].strip()
        split = row['split'].strip().lower()
        row_string = f"{name}|{video}|{start}|{end}|{speaker}|{orth}|{translation}|{split}"

        # Frames are stored in a folder named after the clip's video-id
        frames_path = os.path.join(args.frame_root, split, name)
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

        splits[split].append({
            'fileid': name,
            'folder': os.path.join(split, name, f'*.{args.format}'),
            'signer': speaker,
            'gloss': orth,
            'text': translation,
            'num_frames': n_frames,
            'original_info': row_string,
            'tag': 'phoenix14t'
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

        n_sents = len(set(s['fileid'] for s in samples))
        print(f"  {split_name}: {len(samples)} samples ({n_sents} unique sentences)")

    print(f"\nAnnotation files saved to {args.output_dir}")


if __name__ == '__main__':
    main()
