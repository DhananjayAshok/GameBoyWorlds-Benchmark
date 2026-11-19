import argparse
import glob
import json
import os
from datetime import datetime, timezone

PROGRESS_DIR = "/work/obs/progress"
CLEARS_FILE = "clears.jsonl"


def records(directory):
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
        if os.path.basename(path) == CLEARS_FILE:
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return sorted(out, key=lambda r: r["t"])


def stamp(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%m-%d %H:%M")


def summary(rows):
    runs = {}
    for r in rows:
        runs.setdefault((r["game"], r.get("run") or f"session {r['session']}"), []).append(r)
    print(f"{'game':16} {'run':28} {'last seen':12} {'steps':>8} {'badges':>6} {'maps':>5} {'events':>6}  "
          f"{'subgoals':>8}  last map")
    for (game, run), rs in sorted(runs.items(), key=lambda kv: kv[1][-1]["t"]):
        last = rs[-1]
        best_badges = max(len(r["badges"]) for r in rs)
        best_maps = max(r["maps_reached"] for r in rs)
        best_events = max(r["events"] for r in rs)
        cleared = " CLEARED" if any(r["kind"] == "cleared" for r in rs) else ""
        print(f"{game:16} {run[:28]:28} {stamp(last['t']):12} {last['steps']:>8} {best_badges:>6} {best_maps:>5} "
              f"{best_events:>6}  {last['subgoals_completed']:>3}/{last['subgoals_total']:<4}  {last['map']}{cleared}")


def timeline(rows, run):
    rs = [r for r in rows if r.get("run") == run]
    if not rs:
        print(f"no progress records for run {run!r}")
        return
    for r in rs:
        changes = []
        if r["new_maps"]:
            changes.append("maps: " + ", ".join(r["new_maps"]))
        if r["new_events"]:
            changes.append("events: " + ", ".join(r["new_events"]))
        if r["new_flags"]:
            changes.append("flags: " + ", ".join(r["new_flags"]))
        party = ", ".join(f"{m['species']} {m['level']}" for m in r["party"])
        print(f"{stamp(r['t'])} {r['kind']:8} step {r['steps']:>7}  badges {len(r['badges'])}  maps {r['maps_reached']}"
              f"  events {r['events']}  at {r['map']}  [{party}]")
        for change in changes:
            print(f"    + {change}")


def main():
    parser = argparse.ArgumentParser(description="Progress of every training playthrough, read from game memory "
                                                 "by the server. For judging progress only: the harness must "
                                                 "never read this or any game memory.")
    parser.add_argument("--run", default=None, help="show one run's timeline: what it reached, and when")
    parser.add_argument("--dir", default=PROGRESS_DIR)
    args = parser.parse_args()
    rows = records(args.dir)
    if not rows:
        print(f"no progress records in {args.dir} yet")
        return
    if args.run:
        timeline(rows, args.run)
    else:
        summary(rows)


if __name__ == "__main__":
    main()
