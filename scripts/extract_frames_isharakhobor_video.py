"""
Video-direct replacement for extract_frames_isharakhobor.py.

This script does NOT write frames to disk. It scans CLIPS/, validates that each
CSV sentence maps to a source video, and reports corpus statistics. Use this
instead of frame extraction when running the video-based isharakhobor pipeline.
"""

import argparse
import csv
import os
import sys

sys.path.append('./')

from tqdm import tqdm

from utils.helpers import (
    build_isharakhobor_video_index,
    count_video_frames,
    resolve_isharakhobor_video,
)


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Validate isharakhobor videos for the direct-video pipeline "
            "(no frame extraction)"
        )
    )
    parser.add_argument(
        '--dataset_root', required=True,
        help='Root directory of isharakhobor dataset (contains CLIPS/)',
    )
    parser.add_argument(
        '--csv_path', required=True,
        help='Path to sentence mapping CSV',
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    clips_dir = os.path.join(args.dataset_root, 'CLIPS')
    if not os.path.isdir(clips_dir):
        raise FileNotFoundError(f"CLIPS directory not found: {clips_dir}")

    video_index = build_isharakhobor_video_index(clips_dir)
    print(f"Indexed {len(video_index)} videos under {clips_dir}")

    rows = []
    with open(args.csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} sentences from CSV")

    matched = 0
    missing = 0
    total_frames = 0
    clip_hits = 0
    bounded_hits = 0

    for row in tqdm(rows, desc="Validating videos"):
        video_id = row['video-id'].strip()
        sentence_id = row['sentence-id'].strip()
        rel_path, use_time_bounds = resolve_isharakhobor_video(
            video_index, video_id, sentence_id
        )

        if rel_path is None:
            missing += 1
            continue

        video_path = os.path.join(clips_dir, rel_path)
        n_frames = count_video_frames(
            video_path,
            start_time=row.get('start'),
            end_time=row.get('end'),
            use_time_bounds=use_time_bounds,
        )

        if n_frames == 0:
            missing += 1
            continue

        matched += 1
        total_frames += n_frames
        if use_time_bounds:
            bounded_hits += 1
        else:
            clip_hits += 1

    print(
        f"\nDone! Matched {matched}/{len(rows)} sentences "
        f"({clip_hits} pre-cut clips, {bounded_hits} time-bounded segments)."
    )
    print(f"Estimated total frames (if fully decoded): {total_frames}")
    if missing:
        print(f"Warning: {missing} entries had no readable video.")


if __name__ == '__main__':
    main()
