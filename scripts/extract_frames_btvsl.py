import os
import csv
import cv2
import argparse
from tqdm import tqdm


def extract_frames_from_segment(video_path, start_time, end_time, output_dir, resize=None):
    """Extract frames from a specific time segment of a video file."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"Warning: Invalid FPS for {video_path}")
        cap.release()
        return 0

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = 0
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if resize:
            frame = cv2.resize(frame, resize)
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1
        current_frame += 1

    cap.release()
    return frame_idx


def get_parser():
    parser = argparse.ArgumentParser(
        description="Extract frames from BTVSL dataset videos (cut segments by start/end time)"
    )
    parser.add_argument(
        '--dataset_root', required=True,
        help='Root directory of BTVSL dataset (contains video files like *_cropped_reduced.mp4)'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='Output directory for extracted frames'
    )
    parser.add_argument(
        '--csv_path', required=True,
        help='Path to sentences.csv'
    )
    parser.add_argument(
        '--resize', type=int, nargs=2, default=None,
        help='Resize frames to W H (e.g., --resize 256 256)'
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    rows = []
    with open(args.csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Found {len(rows)} sentences in CSV")

    resize = tuple(args.resize) if args.resize else None
    total_segments = 0
    total_frames = 0
    skipped = 0

    for row in tqdm(rows, desc="Extracting frames"):
        sentence_id = row['sentence_id'].strip()
        video_name = row['video_name'].strip()
        start_time = float(row['start_time'])
        end_time = float(row['end_time'])

        output_subdir = os.path.join(args.output_dir, sentence_id)

        if os.path.exists(output_subdir) and len(os.listdir(output_subdir)) > 0:
            continue

        video_filename = f"{video_name}_cropped_reduced.mp4"
        video_path = os.path.join(args.dataset_root, video_filename)

        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            skipped += 1
            continue

        n_frames = extract_frames_from_segment(
            video_path, start_time, end_time, output_subdir, resize=resize
        )

        if n_frames == 0:
            print(f"Warning: No frames extracted for {sentence_id} "
                  f"(video={video_filename}, t={start_time:.2f}-{end_time:.2f})")
            skipped += 1
            continue

        total_frames += n_frames
        total_segments += 1

    print(f"\nDone! Processed {total_segments} segments, "
          f"extracted {total_frames} frames total. "
          f"Skipped {skipped} entries.")


if __name__ == '__main__':
    main()
