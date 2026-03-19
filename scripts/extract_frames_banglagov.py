import os
import cv2
import csv
import argparse
from tqdm import tqdm


def extract_frames(video_path, output_dir, resize=None):
    """Extract all frames from a video file and save as PNGs."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize:
            frame = cv2.resize(frame, resize)
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1

    cap.release()
    return frame_idx


def get_parser():
    parser = argparse.ArgumentParser(
        description="Extract frames from Banglagov dataset videos"
    )
    parser.add_argument(
        '--dataset_root', required=True,
        help='Root directory of Banglagov dataset (contains Sign_Videos/)'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='Output directory for extracted frames'
    )
    parser.add_argument(
        '--csv_path', required=True,
        help='Path to Bangla_Sign_Sentence_Mapping.csv'
    )
    parser.add_argument(
        '--resize', type=int, nargs=2, default=None,
        help='Resize frames to W H (e.g., --resize 256 256)'
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    sentence_ids = []
    with open(args.csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence_ids.append(str(row['Sentence ID']).strip())

    print(f"Found {len(sentence_ids)} sentences in CSV")

    resize = tuple(args.resize) if args.resize else None
    total_videos = 0
    total_frames = 0
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')

    for sid in tqdm(sentence_ids, desc="Processing sentences"):
        video_dir = os.path.join(args.dataset_root, 'Sign_Videos', sid)

        if not os.path.exists(video_dir):
            print(f"Warning: Video directory not found: {video_dir}")
            continue

        video_files = sorted([
            f for f in os.listdir(video_dir)
            if f.lower().endswith(video_extensions)
        ])

        if len(video_files) == 0:
            print(f"Warning: No video files found in {video_dir}")
            continue

        for signer_idx, video_file in enumerate(video_files):
            video_path = os.path.join(video_dir, video_file)
            output_subdir = os.path.join(args.output_dir, f"{sid}_signer_{signer_idx}")

            if os.path.exists(output_subdir) and len(os.listdir(output_subdir)) > 0:
                continue

            n_frames = extract_frames(video_path, output_subdir, resize=resize)
            total_frames += n_frames
            total_videos += 1

    print(f"Done! Processed {total_videos} videos, extracted {total_frames} frames total.")


if __name__ == '__main__':
    main()
