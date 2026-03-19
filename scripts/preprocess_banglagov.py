import os
import csv
import random
import numpy as np
import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess Banglagov dataset: CSV -> annotation npy files"
    )
    parser.add_argument(
        '--csv_path', required=True,
        help='Path to Bangla_Sign_Sentence_Mapping.csv'
    )
    parser.add_argument(
        '--frame_root', required=True,
        help='Root directory containing extracted frames (output of extract_frames_banglagov.py)'
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

    sentences = []
    with open(args.csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentences.append({
                'sentence_id': str(row['Sentence ID']).strip(),
                'text': row['Natural Sentence'].strip(),
                'gloss': row['Sign Sentence'].strip(),
            })

    print(f"Loaded {len(sentences)} sentences from CSV")

    all_samples = []
    for sent in sentences:
        sid = sent['sentence_id']
        signer_dirs = sorted([
            d for d in os.listdir(args.frame_root)
            if d.startswith(f"{sid}_signer_")
        ])

        for signer_dir in signer_dirs:
            frames_path = os.path.join(args.frame_root, signer_dir)
            n_frames = len([
                f for f in os.listdir(frames_path)
                if f.endswith('.png')
            ])

            if n_frames == 0:
                print(f"Warning: No frames found in {frames_path}, skipping")
                continue

            all_samples.append({
                'fileid': signer_dir,
                'folder': os.path.join(signer_dir, '*.png'),
                'text': sent['text'],
                'gloss': sent['gloss'],
                'sentence_id': sid,
                'num_frames': n_frames,
            })

    print(f"Total samples (sentence x signer): {len(all_samples)}")

    unique_sids = sorted(set(s['sentence_id'] for s in all_samples))
    print(f"Unique sentences: {len(unique_sids)}")

    random.seed(args.seed)
    random.shuffle(unique_sids)

    n_train = int(len(unique_sids) * args.train_ratio)
    n_dev = int(len(unique_sids) * args.dev_ratio)

    train_sids = set(unique_sids[:n_train])
    dev_sids = set(unique_sids[n_train:n_train + n_dev])
    test_sids = set(unique_sids[n_train + n_dev:])

    splits = {
        'train': [s for s in all_samples if s['sentence_id'] in train_sids],
        'dev': [s for s in all_samples if s['sentence_id'] in dev_sids],
        'test': [s for s in all_samples if s['sentence_id'] in test_sids],
    }

    for split_name, samples in splits.items():
        data = {}
        for i, sample in enumerate(samples):
            data[i] = sample

        np.save(os.path.join(args.output_dir, f'{split_name}_info.npy'), data)
        np.save(os.path.join(args.output_dir, f'{split_name}_info_ml.npy'), data)

        n_sents = len(set(s['sentence_id'] for s in samples))
        print(f"  {split_name}: {len(samples)} samples ({n_sents} sentences)")

    print(f"\nAnnotation files saved to {args.output_dir}")


if __name__ == '__main__':
    main()
