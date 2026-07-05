import argparse
import csv
import glob
import json
import os
import sys

HEADERS = ["video-id", "sentence-id", "start", "end", "split", "sentence"]


def main(json_dir, output_csv):
    rows = []
    for path in sorted(glob.glob(os.path.join(json_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for video_key, video_data in data.items():
            video_id = os.path.splitext(video_key)[0]
            for instance in video_data.get("instances", []):
                start = instance.get("start")
                end = instance.get("end")
                rows.append({
                    "video-id": video_id,
                    "sentence-id": f"{video_id}_{start}_{end}",
                    "start": start,
                    "end": end,
                    "split": instance.get("split"),
                    "sentence": instance.get("sentence"),
                })

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a sentence-mapping CSV from a directory of JSON files."
    )
    parser.add_argument(
        "-j",
        "--json",
        dest="json_dir",
        default=None,
        help="Path to the directory containing the JSON files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_csv",
        default=None,
        help=(
            "Path to the output CSV file. "
            "Defaults to 'sentence-mapping.csv' in the parent directory of the JSON path."
        ),
    )
    args = parser.parse_args()

    if not args.json_dir:
        print("Error: no JSON directory path provided.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(args.json_dir):
        print(
            f"Error: '{args.json_dir}' is not a valid directory.", file=sys.stderr
        )
        sys.exit(1)

    output_csv = args.output_csv
    if not output_csv:
        parent_dir = os.path.dirname(os.path.abspath(args.json_dir.rstrip("/")))
        output_csv = os.path.join(parent_dir, "sentence-mapping.csv")

    main(args.json_dir, output_csv)
