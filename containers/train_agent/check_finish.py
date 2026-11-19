import json
import os
import sys

CLEARS = "/work/obs/progress/clears.jsonl"


def main():
    games = [g for g in sys.argv[1:] if g]
    cleared = {}
    if os.path.exists(CLEARS):
        with open(CLEARS) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    cleared.setdefault(record["game"], []).append(record.get("run") or record["session"])
    missing = [g for g in games if g not in cleared]
    for game in games:
        runs = cleared.get(game)
        print(f"{game}: " + (f"cleared by {', '.join(runs)}" if runs else "never cleared"))
    sys.exit(1 if missing or not games else 0)


if __name__ == "__main__":
    main()
