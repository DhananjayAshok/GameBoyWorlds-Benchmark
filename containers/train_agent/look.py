import argparse
import os
import sys

import cyclelib as lib


def main():
    parser = argparse.ArgumentParser(description="Show images to the model and ask a question about them. This is "
                                                 "how you see the screen: frame sheets from digest.py, or any PNG.")
    parser.add_argument("--images", nargs="+", required=True, help="PNG files, at most 8")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    missing = [path for path in args.images if not os.path.exists(path)]
    if missing:
        sys.exit(f"no such file: {missing}")
    if len(args.images) > 8:
        sys.exit("at most 8 images per question")
    print(lib.look(args.images, args.question))


if __name__ == "__main__":
    main()
