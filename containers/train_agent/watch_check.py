import argparse
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import progress

CHECKS_FILE = "/work/CHECKS.md"
CONDITION_RE = re.compile(r"^- \*\*Next condition\*\*: (.*)$")
FALLBACK_RE = re.compile(r"\b(or|and/or)\b", re.IGNORECASE)


def previous_condition(path, run):
    if not os.path.exists(path):
        return None
    current = None
    condition = None
    with open(path) as f:
        for line in f:
            if line.startswith("## "):
                current = line.split("`")[1] if "`" in line else None
            match = CONDITION_RE.match(line.rstrip("\n"))
            if match and current == run:
                condition = match.group(1)
    return condition


def status(rows, run):
    rs = [r for r in rows if r.get("run") == run]
    if not rs:
        return None, None
    return progress.stall(rs), rs[-1]


def show(args, rows):
    stalled, last = status(rows, args.run)
    if last is None:
        print(f"no progress records for run {args.run!r}")
        return
    print(f"run {args.run}: last seen at {last['map']}, step {last['steps']}")
    if stalled:
        print("\n".join(stalled[0]))
    condition = previous_condition(args.checks, args.run)
    print(f"condition set at the last check: {condition or 'none yet'}")


def record(args, rows):
    stalled, last = status(rows, args.run)
    if last is None:
        sys.exit(f"no progress records for run {args.run!r}; start the run first")
    lines, minutes = stalled if stalled else ([], 0)
    problems = []
    if args.verdict == "in_progress" and minutes >= progress.STALL_MINUTES and not args.advancing:
        problems.append(f"the run is STALLED ({minutes} min with no new badge, map or event), so the verdict is "
                        f"stuck; say in_progress only with --advancing naming what it is doing that the counter "
                        f"cannot see, and the evidence that it is advancing")
    if FALLBACK_RE.search(args.condition):
        problems.append("the next condition must be one thing, with no 'or' fallback")
    if not args.evidence.strip():
        problems.append("--evidence is empty: say what you looked at since the last check and what changed")
    if problems:
        print("check not recorded:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(2)
    previous = previous_condition(args.checks, args.run)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    verdict = "In progress" if args.verdict == "in_progress" else "Stuck"
    entry = [f"## `{args.run}` check, {now} UTC", "",
             f"- **Server**: at {last['map']}, step {last['steps']}"]
    entry += [f"  - {line}" for line in lines]
    entry += [f"- **Previous condition**: {previous or 'none'}",
              f"- **Verdict**: {verdict}",
              f"- **Evidence**: {args.evidence}"]
    if args.advancing:
        entry.append(f"- **Advancing, though stalled**: {args.advancing}")
    entry += [f"- **Insights and potential fixes**: {args.insights or 'none'}",
              f"- **Next condition**: {args.condition}", "", ""]
    with open(args.checks, "a") as f:
        f.write("\n".join(entry))
    print(f"recorded in {args.checks}: {verdict}")
    if args.verdict == "stuck":
        print("stuck: stop the run, diagnose this stall from the episode reports and video since the last advance, "
              "write the diagnosis in DEVLOG.md, and fix it before restarting")


def main():
    parser = argparse.ArgumentParser(description="Record one check of a running playthrough in /work/CHECKS.md. "
                                                 "With only --run, show the run's state and the condition set "
                                                 "at the last check.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--verdict", choices=["in_progress", "stuck"])
    parser.add_argument("--evidence", default="", help="what you looked at since the last check, what changed")
    parser.add_argument("--insights", default="", help="insights and potential fixes, with the kind of revision")
    parser.add_argument("--condition", default="", help="one thing the run must do before the next check")
    parser.add_argument("--advancing", default="",
                        help="only for in_progress on a stalled run: what it is doing that the counter cannot see")
    parser.add_argument("--dir", default=progress.PROGRESS_DIR)
    parser.add_argument("--checks", default=CHECKS_FILE)
    args = parser.parse_args()
    rows = progress.records(args.dir)
    if args.verdict is None:
        show(args, rows)
        return
    if not args.condition.strip():
        parser.error("--condition is required with --verdict")
    record(args, rows)


if __name__ == "__main__":
    main()
