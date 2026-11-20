import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agnostic_check
import progress

CHECKS_FILE = "/work/CHECKS.md"
CONDITION_RE = re.compile(r"^- \*\*Next condition\*\*: (.*)$")
SERVER_RE = re.compile(r"^- \*\*Server\*\*: at \S+, step (\d+)$")
VERDICT_RE = re.compile(r"^- \*\*Verdict\*\*: (.*)$")
HARNESS_RE = re.compile(r"^- \*\*Harness\*\*: (\S+)")
EXPECT_RE = re.compile(r"^- \*\*Expect\*\*: (\S+)")
HEADER_TIME_RE = re.compile(r"check, (\d{4}-\d\d-\d\d \d\d:\d\d) UTC")
EXPECT_KINDS = ("map", "event", "badge", "none")
HARNESS_DIR = "/work/harness"
FALLBACK_RE = re.compile(r"\b(or|and/or)\b", re.IGNORECASE)
EPISODE_RE = re.compile(r"\bepisodes?\b", re.IGNORECASE)
MAX_ADVANCING_CHECKS = 3
PROMPT_ONLY_RESTARTS = 2
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
                current = {"step": None, "verdict": None, "condition": None, "advancing": False, "harness": None,
                           "expect": None, "t": None}
                if match := HEADER_TIME_RE.search(line):
                    current["t"] = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M").replace(
                        tzinfo=timezone.utc).timestamp()
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
            elif match := HARNESS_RE.match(line):
                current["harness"] = match.group(1)
            elif match := EXPECT_RE.match(line):
                current["expect"] = match.group(1)
            elif line.startswith("- **Advancing, though stalled**"):
                current["advancing"] = True
    return checks


def harness_commit(directory):
    try:
        head = subprocess.run(["git", "-C", directory, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", directory, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("+uncommitted" if dirty else "")


def parse_expect(text):
    kind, _, name = text.partition(":")
    if kind not in EXPECT_KINDS or (kind in ("map", "event") and not name) or (kind in ("badge", "none") and name):
        return None
    return kind, name


def judge_expect(expect, since, rs):
    parsed = parse_expect(expect or "")
    if parsed is None or parsed[0] == "none" or since is None:
        return None, "nothing to judge"
    kind, name = parsed
    after = [r for r in rs if r["t"] > since and r["kind"] not in progress.SETUP_KINDS]
    if not after:
        return None, "no progress record since that check yet (they are written at each save)"
    if kind == "map":
        met = any(r["map"] == name or name in r["new_maps"] for r in after)
    elif kind == "event":
        met = any(name in r["new_events"] for r in after)
    else:
        before = [len(r["badges"]) for r in rs if r["t"] <= since]
        met = max(len(r["badges"]) for r in after) > (max(before) if before else 0)
    return met, "met" if met else "not met"


def server_change(since, rs):
    after = [r for r in rs if r["t"] > since and r["kind"] not in progress.SETUP_KINDS]
    before = [r for r in rs if r["t"] <= since and r["kind"] not in progress.SETUP_KINDS]
    tiles = sum(r.get("new_tiles", 0) for r in after)
    levels_before = sum(m["level"] or 0 for m in before[-1]["party"]) if before else 0
    levels_after = sum(m["level"] or 0 for m in after[-1]["party"]) if after else levels_before
    return tiles, levels_after - levels_before


def changed_files(directory, old, new):
    try:
        out = subprocess.run(["git", "-C", directory, "diff", "--name-only", old, new], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line for line in out.splitlines() if line]


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
    checks = previous_checks(args.checks, args.run)
    if not checks:
        print("condition set at the last check: none yet")
        return
    previous = checks[-1]
    rs = [r for r in rows if r.get("run") == args.run]
    _, how = judge_expect(previous["expect"], previous["t"], rs)
    print(f"condition set at the last check: {previous['condition']}")
    print(f"its expectation: {previous['expect'] or 'none'} ({how})")


def record(args, rows):
    stalled, last = status(rows, args.run)
    if last is None:
        sys.exit(f"no progress records for run {args.run!r}; start the run first")
    lines, minutes = stalled if stalled else ([], 0)
    checks = previous_checks(args.checks, args.run)
    judged = [c for c in checks if c["verdict"] != VERDICTS["restarted"]]
    rs = [r for r in rows if r.get("run") == args.run]
    problems = []
    if parse_expect(args.expect) is None:
        problems.append("--expect must be map:<MAP>, event:<EVENT>, badge, or none, naming what the next condition "
                        "will show in the progress records (map and event names as progress.py prints them); none "
                        "only when the condition cannot show there, such as winning a battle")
    nones = 0
    for check in reversed(checks):
        if check["expect"] != "none":
            break
        nones += 1
    if args.expect == "none" and nones >= MAX_ADVANCING_CHECKS:
        problems.append(f"--expect none was used at the last {nones} checks; name a map, event or badge the run "
                        f"should reach next")
    met, how = judge_expect(checks[-1]["expect"], checks[-1]["t"], rs) if checks else (None, "")
    if args.verdict == "in_progress" and met is False:
        problems.append(f"the last check expected {checks[-1]['expect']}, and the progress records since then do "
                        f"not show it: the previous condition was not met, so the verdict is stuck")
    if args.verdict == "in_progress":
        if minutes >= progress.STALL_MINUTES and not args.advancing:
            problems.append(f"the run is STALLED ({minutes} min with no new badge, map, event or {progress.TILE_ADVANCE} "
                            f"new tiles), so the verdict "
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
        if args.advancing and checks and checks[-1]["t"] is not None:
            tiles, levels = server_change(checks[-1]["t"], rs)
            if tiles <= 0 and levels <= 0:
                problems.append("--advancing needs the server to show something since the last check, and it "
                                "shows no new tiles stood on and no levels gained; squares on the harness's own "
                                "map are what the harness believes, not where the player has been: the verdict "
                                "is stuck")
    if args.advancing and args.verdict != "in_progress":
        problems.append("--advancing only goes with --verdict in_progress")
    commit = harness_commit(args.harness)
    if args.verdict == "restarted":
        before = checks[-1]["harness"] if checks else None
        if commit is None or commit.endswith("+uncommitted"):
            problems.append("restarted needs the fix committed in /work/harness first, so the run is tied to it")
        elif before == commit:
            problems.append(f"the harness is still at {commit}, as at the last check: restarted is for the check "
                            f"right after a restart with a new fix; without one, judge the run (stuck or "
                            f"in_progress)")
        advance_t = progress.last_advance(rs)[0]["t"] if [r for r in rs if r["kind"] not in progress.SETUP_KINDS] \
            else 0
        this_stall = [c for c in checks if c["verdict"] == VERDICTS["restarted"] and (c["t"] or 0) > advance_t]
        if before and commit and len(this_stall) >= PROMPT_ONLY_RESTARTS:
            files = changed_files(args.harness, before.split("+")[0], commit.split("+")[0])
            if files and all(os.path.basename(path) == "prompts.py" for path in files):
                problems.append(f"the run has been restarted {len(this_stall)} times in this stall (since its last "
                                f"advance) and this fix again changes only prompt text ({', '.join(files)}); "
                                f"rewording has not got it moving, so change what the harness does: its "
                                f"navigation, perception, memory or executors (the second or third kind of "
                                f"revision)")
        try:
            named = agnostic_check.hits(args.harness)
        except (OSError, subprocess.CalledProcessError):
            named = []
        if named:
            listed = "; ".join(f"{path}:{number} ({word})" for path, number, word, _ in named[:8])
            problems.append(f"the harness code names places or people from the training games, so it only works "
                            f"on them: {listed}. Make each generic and commit (run "
                            f"/opt/train/agnostic_check.py to list them)")
    if FALLBACK_RE.search(args.condition):
        problems.append("the next condition must be one thing, with no 'or' fallback")
    if EPISODE_RE.search(args.condition):
        problems.append("the next condition must be something the player does in the game (enter a map, win a "
                        "battle, leave a building), not an episode finishing: episodes finish whether or not the "
                        "run gets anywhere")
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
    entry.append(f"- **Harness**: {commit or 'unknown'}")
    entry += [f"- **Previous condition**: {previous or 'none'}"]
    if checks and checks[-1]["expect"]:
        entry.append(f"  - expected {checks[-1]['expect']}: {how}")
    entry += [f"- **Verdict**: {verdict}",
              f"- **Evidence**: {args.evidence}"]
    if args.advancing:
        entry.append(f"- **Advancing, though stalled**: {args.advancing}")
    entry += [f"- **Insights and potential fixes**: {args.insights or 'none'}",
              f"- **Next condition**: {args.condition}",
              f"- **Expect**: {args.expect}", "", ""]
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
    parser.add_argument("--expect", default="",
                        help="what the next condition will show in the progress records: map:<MAP>, "
                             "event:<EVENT>, badge, or none; the next check is judged by it")
    parser.add_argument("--advancing", default="",
                        help="only for in_progress on a stalled run: what it is doing that the counter cannot see")
    parser.add_argument("--dir", default=progress.PROGRESS_DIR)
    parser.add_argument("--checks", default=CHECKS_FILE)
    parser.add_argument("--harness", default=HARNESS_DIR)
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
