import base64
import glob
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("TRAIN_WORK", "/work")
HARNESS = os.path.join(WORK, "harness")
CYCLE = os.path.join(WORK, "cycle")
PLAYS = os.path.join(WORK, "playthroughs")
TESTS = os.path.join(WORK, "tests")
ITERATIONS = os.path.join(WORK, "iterations")
OBS = os.path.join(WORK, "obs")
PROGRESS = os.path.join(OBS, "progress")
STATE = os.path.join(CYCLE, "state.json")
CONTEXT = os.path.join(CYCLE, "context.json")
DRAFTS = os.path.join(CYCLE, "drafts")
STAGED = os.path.join(CYCLE, "staged")
REQUESTS = os.path.join(CYCLE, "requests")
REPLIES = os.path.join(CYCLE, "replies")
DELETED = os.path.join(CYCLE, "deleted_checkpoints.jsonl")
TAXONOMY = os.path.join(HERE, "taxonomy.md")
PYTHON = "/opt/harness/bin/python"

CHECK_SECONDS = int(os.environ.get("TRAIN_CHECK_SECONDS", 300))
LONG_SECONDS = int(os.environ.get("TRAIN_LONG_SECONDS", 3 * 3600))
SHORT_TEST_MINUTES = 10
SHORT_CHECK_SECONDS = int(os.environ.get("TRAIN_SHORT_CHECK_SECONDS", 60))
TESTS_PER_ISSUE = 3
ISSUES_PER_CYCLE = int(os.environ.get("TRAIN_ISSUES_PER_CYCLE", 5))
BIG_ISSUES_PER_CYCLE = int(os.environ.get("TRAIN_BIG_ISSUES_PER_CYCLE", 3))
KEEP_FRACTION = 0.1
SETUP_KINDS = ("reset", "load", "close")
LEVELS = ("high", "medium", "low")
MIN_TEXT = 40
BEGINNING = "beginning"

SCREEN_QUESTION = (
    "These images are sheets of frames, in time order (numbered), from a Game Boy role-playing game being played "
    "by a program. Describe what the player does over these frames: where it is, how it moves, which menus, "
    "conversations and battles appear and how they end, and any behaviour that repeats. Report only what the "
    "frames show. Do not guess at what happens between them.")


def stamp(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%m-%d %H:%M:%S")


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(value, f, indent=1)
    os.replace(path + ".tmp", path)


def read_jsonl(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    return rows


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(path + ".tmp", path)


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


def git(*args, check=True):
    out = subprocess.run(["git", "-C", HARNESS, *args], capture_output=True, text=True)
    if check and out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {out.stderr.strip()}")
    return out.stdout.strip()


def head():
    return git("rev-parse", "HEAD")


def dirty():
    return bool(git("status", "--porcelain"))


def commits_between(start, end="HEAD"):
    return [line for line in git("rev-list", "--reverse", f"{start}..{end}").splitlines() if line]


def taxonomy():
    with open(TAXONOMY) as f:
        text = f.read()
    locus_part, _, code_part = text.partition("## Codes")
    loci = set(re.findall(r"^\| `([a-z\-/]+)` \|", locus_part, flags=re.M))
    codes = set(re.findall(r"^\*\*`([a-z\-/]+)`\*\*", code_part, flags=re.M))
    return loci, codes


def run_dir(run):
    for root in (PLAYS, TESTS):
        path = os.path.join(root, run)
        if os.path.isdir(path):
            return path
    return None


def run_info(run):
    directory = run_dir(run)
    if directory is None:
        return None
    return read_json(os.path.join(directory, "playthrough.json")) or read_json(os.path.join(directory, "test.json"))


def checkpoints(run):
    directory = run_dir(run)
    latest = {}
    for row in read_jsonl(os.path.join(directory, "harness", "checkpoints.jsonl")) if directory else []:
        latest[row["id"]] = row
    return [latest[key] for key in sorted(latest)]


def deleted_saves():
    return {row["server_save"] for row in read_jsonl(DELETED)}


def find_checkpoint(ref):
    run, _, checkpoint = str(ref).rpartition(":")
    if not run or not checkpoint.isdigit():
        return None, f"{ref!r} is not <run>:<id>"
    info = run_info(run)
    if info is None:
        return None, f"there is no run {run!r}"
    for row in checkpoints(run):
        if row["id"] == int(checkpoint):
            if row["server_save"] in deleted_saves():
                return None, f"checkpoint {ref} was deleted"
            return dict(row, game=info["game"], ref=f"{run}:{row['id']}"), None
    return None, f"run {run!r} has no checkpoint {checkpoint}"


def progress_records(run):
    rows = []
    for path in sorted(glob.glob(os.path.join(PROGRESS, "*.jsonl"))):
        if os.path.basename(path) != "clears.jsonl":
            rows += [r for r in read_jsonl(path) if r.get("run") == run]
    return sorted(rows, key=lambda r: r["t"])


def cleared_games():
    return {row["game"] for row in read_jsonl(os.path.join(PROGRESS, "clears.jsonl"))}


def signal(run, since, now=None):
    now = now or time.time()
    played = [r for r in progress_records(run) if r["kind"] not in SETUP_KINDS]
    if not played:
        return {"recommendation": "unclear", "text": "The server has no progress records for this run yet."}

    def over(seconds):
        rows = [r for r in played if r["t"] > now - seconds]
        return (sum(r.get("new_tiles", 0) for r in rows), sorted({m for r in rows for m in r.get("new_maps", [])}),
                sum(len(r.get("new_events", [])) for r in rows))

    window = [r for r in played if r["t"] > since]
    last = played[-1]
    fresh = [r for r in played if r.get("new_tiles", 0) > 0 or r.get("new_maps")]
    quiet_from = fresh[-1] if fresh else played[0]
    quiet_minutes = (now - quiet_from["t"]) / 60
    tiles_15, maps_15, events_15 = over(15 * 60)
    tiles_30, maps_30, events_30 = over(30 * 60)
    minutes_played = (now - played[0]["t"]) / 60
    badges = len(last.get("badges", []))
    if minutes_played < 15:
        recommendation, why = "unclear", "fewer than 15 minutes of play to judge by"
    elif not maps_15 and not events_15 and tiles_15 < 10:
        recommendation = "stuck"
        why = f"no new map or story event and only {tiles_15} new grid spots in the last 15 minutes"
    elif minutes_played >= 30 and not maps_30 and not events_30 and tiles_30 < 25:
        recommendation = "stuck"
        why = f"no new map or story event and only {tiles_30} new grid spots in the last 30 minutes"
    else:
        recommendation = "progressing"
        why = (f"{tiles_15} new grid spots, {len(maps_15)} new map(s) and {events_15} new story event(s) in the "
               f"last 15 minutes")
    lines = [
        f"Now at: {last.get('map')}, step {last['steps']}, {badges} badge(s), {last.get('maps_reached')} maps "
        f"reached, {last.get('tiles')} grid spots stood on in all.",
        f"Since the last check: {sum(r.get('new_tiles', 0) for r in window)} new grid spots, new maps: "
        f"{sorted({m for r in window for m in r.get('new_maps', [])}) or 'none'}.",
        f"Last 15 minutes: {tiles_15} new grid spots, new maps: {maps_15 or 'none'}.",
        f"Last 30 minutes: {tiles_30} new grid spots, new maps: {maps_30 or 'none'}.",
        f"Time since the player last stood on a new grid spot or entered a new map: {quiet_minutes:.0f} minutes "
        f"({last['steps'] - quiet_from['steps']} steps).",
        f"Party: {', '.join(str(m.get('species')) + ' ' + str(m.get('level')) for m in last.get('party', [])) or 'none'}.",
        f"CONTROLLER RECOMMENDATION: {recommendation.upper()} ({why}).",
    ]
    return {"recommendation": recommendation, "why": why, "text": "\n".join(lines), "map": last.get("map"),
            "steps": last["steps"], "tiles": last.get("tiles"), "maps_reached": last.get("maps_reached"),
            "badges": badges, "new_tiles_window": sum(r.get("new_tiles", 0) for r in window),
            "new_tiles_15": tiles_15, "new_tiles_30": tiles_30, "quiet_minutes": round(quiet_minutes, 1)}


def videos(run, since, until):
    sessions = {r["session"] for r in progress_records(run)}
    found = []
    for session in sessions:
        for path in glob.glob(os.path.join(OBS, session, "*", "videos", "*.mp4")):
            modified = os.path.getmtime(path)
            if since < modified <= until:
                found.append((modified, path))
    return [path for _, path in sorted(found)]


def frame_sheets(run, since, until, out_dir, label, frames=12, per_sheet=6):
    import cv2
    import numpy as np
    readable = []
    for path in videos(run, since, until):
        capture = cv2.VideoCapture(path)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if count > 0:
            readable.append((path, count))
    total = sum(count for _, count in readable)
    if total == 0:
        return []
    wanted = [int(i * (total - 1) / max(1, frames - 1)) for i in range(min(frames, total))]
    picked = []
    offset = 0
    for path, count in readable:
        capture = cv2.VideoCapture(path)
        for index in [w - offset for w in wanted if offset <= w < offset + count]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok:
                picked.append(frame)
        capture.release()
        offset += count
    os.makedirs(out_dir, exist_ok=True)
    sheets = []
    for start in range(0, len(picked), per_sheet):
        tiles = []
        for number, frame in enumerate(picked[start:start + per_sheet], start=start + 1):
            frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
            frame = cv2.copyMakeBorder(frame, 22, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(frame, f"#{number}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(frame)
        while len(tiles) % 3:
            tiles.append(np.full_like(tiles[0], 255))
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
        path = os.path.join(out_dir, f"{label}_{start // per_sheet + 1}.png")
        cv2.imwrite(path, np.vstack(rows))
        sheets.append(path)
    return sheets


def look(images, question, max_tokens=1200):
    from openai import OpenAI
    client = OpenAI(base_url=os.environ["MODEL_GATEWAY_URL"], api_key="none", timeout=120)
    content = [{"type": "text", "text": question}]
    for path in images:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
    reply = client.chat.completions.create(model=os.environ["SANDBOX_MODEL"], max_tokens=max_tokens,
                                           messages=[{"role": "user", "content": content}])
    return (reply.choices[0].message.content or "").strip()


def trajectory(run, since, until):
    directory = run_dir(run)
    rows = read_jsonl(os.path.join(directory, "harness", "trajectory.jsonl")) if directory else []
    return [r for r in rows if since < r["t"] <= until]


def digest(run, since, until, label, describe=True, text_limit=700):
    directory = run_dir(run)
    lines = [f"# {run}: {stamp(since)} to {stamp(until)} UTC", "",
             "## What the server measured (from game memory; for judging only, never for the harness)", "",
             signal(run, since, until)["text"], ""]
    records = [r for r in progress_records(run) if since < r["t"] <= until]
    for r in records:
        new = f" new maps: {', '.join(r['new_maps'])}" if r.get("new_maps") else ""
        lines.append(f"- {stamp(r['t'])} {r['kind']:8} step {r['steps']} at {r.get('map')} "
                     f"tiles {r.get('tiles')} (+{r.get('new_tiles', 0)}){new}")
    lines += ["", "## What the harness intended, planned, tried and concluded (its own record; it can be wrong)", ""]
    events = trajectory(run, since, until)
    if not events:
        lines.append("No trajectory events in this window: no episode started or finished, or the harness has "
                     "stopped writing trajectory.jsonl, which is itself a harness problem.")
    for e in events:
        text = e["text"] if len(e["text"]) <= text_limit else e["text"][:text_limit] + " [...]"
        lines.append(f"- {stamp(e['t'])} episode {e['episode']} {e['layer']} {e['kind']} (step {e.get('steps')}): "
                     + text.replace("\n", "\n    "))
    lines += ["", "## What the screen showed", ""]
    sheets = frame_sheets(run, since, until, os.path.join(directory, "frames"), label) if directory else []
    if not sheets:
        lines.append("No finished video in this window (a video file closes when an episode ends).")
    else:
        lines.append("Frame sheets, in time order: " + ", ".join(sheets))
        if describe:
            try:
                lines += ["", "Described from those frames by the model (look.py asks follow-up questions):", "",
                          look(sheets, SCREEN_QUESTION)]
            except Exception as error:
                lines += ["", f"The frames could not be described ({type(error).__name__}: {error}); "
                              "use look.py on the sheets yourself."]
    return "\n".join(lines) + "\n"


def need_text(errors, value, name, minimum=MIN_TEXT):
    if not isinstance(value, str) or len(value.strip()) < minimum:
        errors.append(f"{name} must be text of at least {minimum} characters")


def check_issue_codes(errors, issue, name):
    loci, codes = taxonomy()
    if issue.get("locus") not in loci:
        errors.append(f"{name}: locus {issue.get('locus')!r} is not in taxonomy.md")
    code = issue.get("code")
    if not isinstance(code, str) or (code not in codes and not re.fullmatch(r"new:[a-z\-]+/[a-z\-]+", code)):
        errors.append(f"{name}: code {code!r} is not in taxonomy.md and is not new:<group>/<name>")


def validate_start(draft, context):
    errors = []
    if not isinstance(draft, dict):
        return ["the draft must be a JSON object"]
    need_text(errors, draft.get("why"), "why")
    start = draft.get("from")
    if start != BEGINNING:
        checkpoint, problem = find_checkpoint(start)
        if problem:
            errors.append(f"from: {problem}; use \"{BEGINNING}\" or one of the checkpoints listed in the prompt")
        elif checkpoint["game"] != context["game"]:
            errors.append(f"from: {start} is a {checkpoint['game']} checkpoint and this playthrough is "
                          f"{context['game']}")
    return errors


def validate_check(draft, context):
    errors = []
    if not isinstance(draft, dict):
        return ["the draft must be a JSON object"]
    summary = draft.get("summary") if isinstance(draft.get("summary"), dict) else {}
    for part in ("intentions", "plans", "attempts", "on_screen"):
        need_text(errors, summary.get(part), f"summary.{part}")
    issues = draft.get("issues")
    if not isinstance(issues, list):
        errors.append("issues must be a list (empty if you saw none)")
    for i, issue in enumerate(issues if isinstance(issues, list) else []):
        if not isinstance(issue, dict):
            errors.append(f"issues[{i}] must be an object")
            continue
        check_issue_codes(errors, issue, f"issues[{i}]")
        for part in ("observation", "cause", "correct_behavior"):
            need_text(errors, issue.get(part), f"issues[{i}].{part}")
    stuck = draft.get("stuck") if isinstance(draft.get("stuck"), dict) else {}
    if stuck.get("verdict") not in ("stuck", "progressing"):
        errors.append("stuck.verdict must be \"stuck\" or \"progressing\"")
    need_text(errors, stuck.get("reasoning"), "stuck.reasoning")
    recommendation = context.get("recommendation")
    if recommendation in ("stuck", "progressing") and stuck.get("verdict") in ("stuck", "progressing") \
            and stuck["verdict"] != recommendation:
        need_text(errors, stuck.get("disagreement"),
                  f"stuck.disagreement (the controller recommends {recommendation}; say what you see that its "
                  "counters do not, with evidence)")
    return errors


def candidates(pid):
    seen = []
    for row in read_jsonl(os.path.join(PLAYS, pid, "checks.jsonl")):
        ref = (row.get("checkpoint") or {}).get("ref")
        if ref and ref not in seen:
            seen.append(ref)
    return seen


def keep_limit(pid):
    return int(KEEP_FRACTION * len(candidates(pid)))


def validate_saves(draft, context):
    errors = []
    keep = draft.get("keep") if isinstance(draft, dict) else None
    if not isinstance(keep, list):
        return ["keep must be a list (empty to keep none)"]
    allowed = candidates(context["pid"])
    limit = keep_limit(context["pid"])
    if len(keep) > limit:
        errors.append(f"you may keep at most {limit} of the {len(allowed)} checkpoints; you listed {len(keep)}")
    seen = set()
    for i, row in enumerate(keep):
        if not isinstance(row, dict) or row.get("id") not in allowed:
            errors.append(f"keep[{i}].id must be one of the checkpoints the checks recorded")
            continue
        if row["id"] in seen:
            errors.append(f"keep[{i}].id {row['id']} is listed twice")
        seen.add(row["id"])
        need_text(errors, row.get("context"), f"keep[{i}].context")
        need_text(errors, row.get("useful_for"), f"keep[{i}].useful_for")
    return errors


def earlier_issues(pid):
    known = {}
    for other in sorted(os.listdir(PLAYS)) if os.path.isdir(PLAYS) else []:
        if other != pid:
            for name in ("issues.jsonl", "big_issues.jsonl"):
                for row in read_jsonl(os.path.join(PLAYS, other, name)):
                    known[f"{other}:{row['id']}"] = row
    return known


def validate_issue_list(draft, context, prefix, maximum=None, big=False):
    errors = []
    issues = draft.get("issues") if isinstance(draft, dict) else None
    if not isinstance(issues, list):
        return ["issues must be a list"]
    if maximum is not None and len(issues) > maximum:
        errors.append(f"list at most {maximum} issues")
    known = earlier_issues(context["pid"])
    ids, ranks = [], []
    for i, issue in enumerate(issues):
        if not isinstance(issue, dict):
            errors.append(f"issues[{i}] must be an object")
            continue
        name = f"issues[{i}]"
        if not isinstance(issue.get("id"), str) or not re.fullmatch(prefix + r"\d+", issue["id"]):
            errors.append(f"{name}.id must be {prefix}1, {prefix}2, ...")
        ids.append(issue.get("id"))
        ranks.append(issue.get("rank"))
        need_text(errors, issue.get("description"), f"{name}.description")
        need_text(errors, issue.get("where"), f"{name}.where (which checks, times and episodes show it)", 20)
        need_text(errors, issue.get("probable_cause"), f"{name}.probable_cause")
        previous = issue.get("present_in_previous")
        if previous != "no" and previous not in known:
            errors.append(f"{name}.present_in_previous must be \"no\" or an earlier issue as <playthrough>:<id>; "
                          f"known: {sorted(known) or 'none yet'}")
        if big:
            need_text(errors, issue.get("proposed_change"), f"{name}.proposed_change")
        else:
            check_issue_codes(errors, issue, name)
            for part in ("likely_gain", "solvability"):
                if issue.get(part) not in LEVELS:
                    errors.append(f"{name}.{part} must be one of {LEVELS}")
    if len(set(ids)) != len(ids):
        errors.append("issue ids must be unique")
    if sorted(r for r in ranks if isinstance(r, int)) != list(range(1, len(issues) + 1)):
        errors.append(f"rank must number the issues 1 to {len(issues)}, 1 being the one to address first")
    return errors


def validate_issues(draft, context):
    return validate_issue_list(draft, context, "I")


def validate_big_issues(draft, context):
    return validate_issue_list(draft, context, "B", maximum=BIG_ISSUES_PER_CYCLE, big=True)


def validate_summary(draft, context):
    errors = []
    if not isinstance(draft, dict):
        return ["the draft must be a JSON object"]
    need_text(errors, draft.get("summary"), "summary", 400)
    expected = context.get("expectations", [])
    held = draft.get("expectations")
    if not isinstance(held, list) or len(held) != len(expected):
        errors.append(f"expectations must judge each of the {len(expected)} expectation(s) in the prompt, in order")
        return errors
    for i, row in enumerate(held):
        if not isinstance(row, dict) or row.get("held") not in ("yes", "no", "unclear"):
            errors.append(f"expectations[{i}].held must be yes, no or unclear")
        else:
            need_text(errors, row.get("evidence"), f"expectations[{i}].evidence", 20)
    return errors


def test_result(tid):
    info = read_json(os.path.join(TESTS, tid, "test.json")) or {}
    verdicts = read_jsonl(os.path.join(TESTS, tid, "checks.jsonl"))
    last = verdicts[-1] if verdicts else {}
    if last.get("verdict") == "fixed" or (last.get("verdict") == "stop_point_reached" and last.get("passed") == "yes"):
        return "passed"
    if info.get("status") == "running":
        return "running"
    return "failed"


def investigation_path(context):
    return os.path.join(PLAYS, context["pid"], "investigations", f"{context['issue']['id']}.md")


def validate_fix(draft, context):
    errors = []
    if not isinstance(draft, dict):
        return ["the draft must be a JSON object"]
    issue = context["issue"]
    if draft.get("issue") != issue["id"]:
        errors.append(f"issue must be {issue['id']!r}, the one this step is about")
    status = draft.get("status")
    if status not in ("fixed", "deferred"):
        errors.append("status must be \"fixed\" or \"deferred\"")
    need_text(errors, draft.get("diagnosis"), "diagnosis", 100)
    path = investigation_path(context)
    try:
        with open(path) as f:
            written = f.read()
    except OSError:
        written = ""
    wanted = ["## Diagnosis", "## Evidence", "## Fix"]
    if issue.get("present_in_previous", "no") != "no":
        wanted.append("## Why the earlier fix did not work")
    missing = [section for section in wanted if section not in written]
    if len(written) < 300 or missing:
        errors.append(f"{path} must exist, be at least 300 characters, and have the sections {wanted}"
                      + (f"; missing {missing}" if missing else ""))
    if dirty():
        errors.append("the harness has uncommitted changes; commit the fix (one commit per fix) or discard them")
    tests = context.get("tests", [])
    results = [test_result(tid) for tid in tests]
    made = commits_between(context["start_sha"])
    if status == "fixed":
        if not made:
            errors.append("status is fixed but the harness has no new commit since this step began")
        listed = draft.get("commits")
        if not isinstance(listed, list) or not listed or any(
                not any(full.startswith(str(sha)) for full in made) for sha in listed):
            errors.append(f"commits must list the commit(s) of this fix, from: {[c[:10] for c in made]}")
        if results and results[-1] != "passed":
            errors.append(f"the last short test ({tests[-1]}) did not pass, so this is not fixed: change the fix "
                          f"and test again ({len(tests)} of {TESTS_PER_ISSUE} tests used), or revert your commits "
                          "with git revert and record it as deferred")
        check = subprocess.run([PYTHON, os.path.join(HERE, "agnostic_check.py")], capture_output=True, text=True)
        if check.returncode != 0:
            errors.append("agnostic_check.py found names from the training games in the harness:\n"
                          + (check.stdout + check.stderr).strip()[-1500:])
    if status == "deferred":
        need_text(errors, draft.get("note"), "note (why it is deferred, and what was tried)")
        if git("diff", "--stat", context["start_sha"], "HEAD"):
            errors.append("a deferred issue leaves the harness as it was when this step began: revert your "
                          "commits with git revert")
    return errors


def validate_iteration(draft, context):
    errors = []
    if not isinstance(draft, dict):
        return ["the draft must be a JSON object"]
    need_text(errors, draft.get("summary"), "summary", 200)
    changes = draft.get("changes")
    if not isinstance(changes, list):
        errors.append("changes must be a list")
        changes = []
    listed = " ".join(str(row.get("commit")) for row in changes if isinstance(row, dict))
    for row in context["fixed"]:
        for sha in row["commits"]:
            if sha[:7] not in listed:
                errors.append(f"changes must include commit {sha[:10]} (the fix for {row['issue']})")
    for i, row in enumerate(changes):
        if not isinstance(row, dict):
            errors.append(f"changes[{i}] must be an object")
            continue
        need_text(errors, row.get("what"), f"changes[{i}].what", 20)
    deferred = draft.get("deferred")
    if not isinstance(deferred, list):
        errors.append("deferred must be a list")
        deferred = []
    named = {row.get("issue") for row in deferred if isinstance(row, dict)}
    for issue in context["deferred"]:
        if issue not in named:
            errors.append(f"deferred must include {issue}, with why")
    expectations = draft.get("expectations")
    if not isinstance(expectations, list) or len(expectations) < 2:
        errors.append("expectations must list at least 2 things the next playthroughs should show")
        expectations = []
    for i, row in enumerate(expectations):
        if not isinstance(row, dict) or row.get("game") not in context["games"] + ["both"]:
            errors.append(f"expectations[{i}].game must be one of {context['games'] + ['both']}")
            continue
        need_text(errors, row.get("text"), f"expectations[{i}].text")
    return errors


VALIDATORS = {"start": validate_start, "check": validate_check, "saves": validate_saves,
              "issues": validate_issues, "summary": validate_summary, "fix": validate_fix,
              "big_issues": validate_big_issues, "iteration": validate_iteration}

TEMPLATES = {
    "start": {"from": "beginning | <run>:<checkpoint id>", "why": "one or two sentences"},
    "check": {"summary": {"intentions": "what the strategist was trying to do, citing episodes",
                          "plans": "how the supervisor broke that down",
                          "attempts": "what the executors actually pressed and tried",
                          "on_screen": "what the player did on screen, from the frames, not from the harness's text"},
              "issues": [{"locus": "from taxonomy.md", "code": "from taxonomy.md",
                          "observation": "how the issue can be seen, citing episode and time",
                          "cause": "the likely cause in the harness",
                          "correct_behavior": "what the player should have done instead"}],
              "stuck": {"verdict": "stuck | progressing",
                        "reasoning": "from the grid spots and maps, and from the earlier summaries",
                        "disagreement": "only when your verdict differs from the controller's recommendation"}},
    "saves": {"keep": [{"id": "<run>:<checkpoint id>", "context": "where the player is and what just happened",
                        "useful_for": "what starting here would let you test"}]},
    "issues": {"issues": [{"id": "I1", "locus": "from taxonomy.md", "code": "from taxonomy.md",
                           "description": "what the player agent struggles with",
                           "where": "which checks, times and episodes of this playthrough show it",
                           "probable_cause": "the likely cause in the harness",
                           "present_in_previous": "no | <playthrough>:<issue id>",
                           "likely_gain": "high | medium | low", "solvability": "high | medium | low",
                           "rank": 1}]},
    "summary": {"summary": "one account of the whole playthrough",
                "expectations": [{"held": "yes | no | unclear", "evidence": "what shows it"}]},
    "fix": {"issue": "I1", "status": "fixed | deferred", "diagnosis": "the cause, as you found it",
            "commits": ["<sha>"], "note": "for deferred: why, and what was tried"},
    "big_issues": {"issues": [{"id": "B1", "description": "the larger problem in how the player behaves",
                               "where": "which playthroughs, checks and issues show it",
                               "probable_cause": "what in the harness's design produces it",
                               "proposed_change": "the design change that would address it",
                               "present_in_previous": "no | <playthrough>:<issue id>", "rank": 1}]},
    "iteration": {"summary": "what this iteration changed and why",
                  "changes": [{"commit": "<sha>", "issue": "I1", "what": "what the commit changes"}],
                  "deferred": [{"issue": "I2", "why": "why it was deferred"}],
                  "expectations": [{"game": "pokemon_red | pokemon_crystal | both",
                                    "text": "one checkable thing the next playthrough should show"}]},
}
