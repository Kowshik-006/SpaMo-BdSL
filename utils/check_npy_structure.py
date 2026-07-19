import numpy as np
import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description="Check the structure of the npy file"
    )
    parser.add_argument(
        '--path', required=True,
        help='Path to the npy file'
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    data = np.load(args.path, allow_pickle=True)

    item = data.item()[0]

    for key, value in item.items():
        print(f"{key} : {value}")

if __name__ == "__main__":
    main()