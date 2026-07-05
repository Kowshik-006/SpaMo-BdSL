import os
import cv2
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
        description="Extract frames from isharakhobor dataset videos (CLIPS/<subdir>/*.mp4)"
    )
    parser.add_argument(
        '--dataset_root', required=True,
        help='Root directory of isharakhobor dataset (contains CLIPS/)'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='Output directory for extracted frames'
    )
    parser.add_argument(
        '--resize', type=int, nargs=2, default=None,
        help='Resize frames to W H (e.g., --resize 256 256)'
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    clips_dir = os.path.join(args.dataset_root, 'CLIPS')
    if not os.path.isdir(clips_dir):
        raise FileNotFoundError(f"CLIPS directory not found: {clips_dir}")

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')

    video_paths = []
    for subdir in sorted(os.listdir(clips_dir)):
        subdir_path = os.path.join(clips_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for video_file in sorted(os.listdir(subdir_path)):
            if video_file.lower().endswith(video_extensions):
                video_paths.append(os.path.join(subdir_path, video_file))

    print(f"Found {len(video_paths)} video files in {clips_dir}")

    resize = tuple(args.resize) if args.resize else None
    total_videos = 0
    total_frames = 0
    skipped = 0

    for video_path in tqdm(video_paths, desc="Extracting frames"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_subdir = os.path.join(args.output_dir, video_name)

        if os.path.exists(output_subdir) and len(os.listdir(output_subdir)) > 0:
            continue

        n_frames = extract_frames(video_path, output_subdir, resize=resize)

        if n_frames == 0:
            print(f"Warning: No frames extracted for {video_path}")
            skipped += 1
            continue

        total_frames += n_frames
        total_videos += 1

    print(f"\nDone! Processed {total_videos} videos, "
          f"extracted {total_frames} frames total. "
          f"Skipped {skipped} entries.")


if __name__ == '__main__':
    main()
