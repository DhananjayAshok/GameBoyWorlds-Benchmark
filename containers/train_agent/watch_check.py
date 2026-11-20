import argparse
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import progress

CHECKS_FILE = "/work/CHECKS.md"
CONDITION_RE = re.compile(r"^- \*\*Next condition\*\*: (.*)$")
SERVER_RE = re.compile(r"^- \*\*Server\*\*: at \S+, step (\d+)$")
VERDICT_RE = re.compile(r"^- \*\*Verdict\*\*: (.*)$")
FALLBACK_RE = re.compile(r"\b(or|and/or)\b", re.IGNORECASE)
MAX_ADVANCING_CHECKS = 3
VERDICTS = {"in_progress": "In progress", "stuck": "Stuck", "restarted": "Restarted"}


def previous_checks(path, run):
    if not os.path.exists(path):
        return []
    checks = []
    current = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("## "):
                current = {"step": None, "verdict": None, "condition": None, "advancing": False}
                if "`" in line and line.split("`")[1] == run:
                    checks.append(current)
                continue
            if current is None:
                continue
            if match := SERVER_RE.match(line):
                current["step"] = int(match.group(1))
            elif match := VERDICT_RE.match(line):
                current["verdict"] = match.group(1)
            elif match := CONDITION_RE.match(line):
                current["condition"] = match.group(1)
            elif line.startswith("- **Advancing, though stalled**"):
                current["advancing"] = True
    return checks


def previous_condition(path, run):
    checks = previous_checks(path, run)
    return checks[-1]["condition"] if checks else None


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
    checks = previous_checks(args.checks, args.run)
    judged = [c for c in checks if c["verdict"] != VERDICTS["restarted"]]
    problems = []
    if args.verdict == "in_progress":
        if minutes >= progress.STALL_MINUTES and not args.advancing:
            problems.append(f"the run is STALLED ({minutes} min with no new badge, map or event), so the verdict "
                            f"is stuck; say in_progress only with --advancing naming what it is doing that the "
                            f"counter cannot see, and the evidence that it is advancing")
        if judged and judged[-1]["step"] is not None and last["steps"] <= judged[-1]["step"]:
            problems.append(f"the server's step count has not moved since the last check (step {last['steps']}), "
                            f"so nothing has advanced; the verdict is stuck, or restarted if you have just "
                            f"restarted the run with a fix")
        streak = 0
        for check in reversed(judged):
            if not check["advancing"]:
                break
            streak += 1
        if args.advancing and streak >= MAX_ADVANCING_CHECKS:
            problems.append(f"--advancing was used at the last {streak} checks; a stall the counter still shows "
                            f"after that is a stall: the verdict is stuck")
    if args.advancing and args.verdict != "in_progress":
        problems.append("--advancing only goes with --verdict in_progress")
    if FALLBACK_RE.search(args.condition):
        problems.append("the next condition must be one thing, with no 'or' fallback")
    if not args.evidence.strip():
        problems.append("--evidence is empty: say what you looked at since the last check and what changed")
    if problems:
        print("check not recorded:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(2)
    previous = checks[-1]["condition"] if checks else None
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    verdict = VERDICTS[args.verdict]
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
    parser.add_argument("--verdict", choices=list(VERDICTS),
                        help="restarted: you have just restarted the run with a fix, so there is nothing to judge yet")
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
