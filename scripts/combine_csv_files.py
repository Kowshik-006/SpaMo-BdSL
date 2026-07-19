import sys
from pathlib import Path
import csv
import argparse

split_list = ['train', 'dev', 'test']
SPLIT_KEY = 'split'

def get_parser():
    parser = argparse.ArgumentParser(
        description="Combine multiple CSV files into a single CSV file"
    )
    parser.add_argument(
        '--csv_root', required=True,
        help='Root directory of the csv files (contains *.csv)'
    )
    parser.add_argument(
        '--output_csv', required=True,
        help='Output CSV file'
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    csv_root = Path(args.csv_root)
    output_csv = csv_root / args.output_csv

    rows = []
    for split in split_list:
        csv_files = csv_root.glob(f'*{split}.corpus.csv')
        if not csv_files:
            continue
        for csv_file in csv_files:
            with open(csv_file, 'r',encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f, delimiter='|')
                
                if reader.fieldnames is None:
                    print(f"Error: CSV has no header row: {csv_file}", file=sys.stderr)
                    sys.exit(1)
                
                fieldnames = list(reader.fieldnames)
                # print(fieldnames)
                if SPLIT_KEY not in fieldnames:
                    fieldnames.append(SPLIT_KEY)
                for row in reader:
                    row[SPLIT_KEY] = split
                    rows.append(row)
    
    with open(output_csv, 'w',encoding='utf-8-sig', newline='') as f:
        print(fieldnames)
        writer = csv.DictWriter(f, delimiter='|', fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
                
