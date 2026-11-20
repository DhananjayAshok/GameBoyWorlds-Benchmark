import argparse
import os
import re
import subprocess
import sys

NAMES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "game_names.txt")
HARNESS_DIR = "/work/harness"


def names():
    with open(NAMES_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def pattern():
    return re.compile(r"\b(" + "|".join(re.escape(name) for name in names()) + r")(?:'?s)?\b", re.IGNORECASE)


def tracked_python(directory):
    out = subprocess.run(["git", "-C", directory, "ls-files", "*.py"], capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


def hits(directory):
    found = []
    regex = pattern()
    for path in tracked_python(directory):
        try:
            with open(os.path.join(directory, path), errors="replace") as f:
                for number, line in enumerate(f, 1):
                    for match in regex.finditer(line):
                        found.append((path, number, match.group(0), line.strip()))
        except OSError:
            continue
    return found


def main():
    parser = argparse.ArgumentParser(description="Find names from the training games (towns, gym leaders, "
                                                 "professors, regions) in the harness's Python code. A harness "
                                                 "that names them only works on the games it was trained on.")
    parser.add_argument("--harness", default=HARNESS_DIR)
    args = parser.parse_args()
    found = hits(args.harness)
    if not found:
        print("no training-game names in the harness code")
        return
    print(f"{len(found)} training-game name(s) in the harness code; remove or replace each with a generic one:")
    for path, number, word, line in found:
        print(f"  {path}:{number}: {word}: {line[:160]}")
    sys.exit(1)


if __name__ == "__main__":
    main()
