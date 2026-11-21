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
    played = [r for r in progress_records(run) if r["kind"] not in SETUP_KINDS and r["t"] <= now]
    if not played:
        return {"recommendation": "unclear", "text": "The server has no progress records for this run yet."}

    def over(seconds):
        rows = [r for r in played if r["t"] > now - seconds]
        return (sum(r.get("new_tiles", 0) for r in rows), sorted({m for r in rows for m in r.get("new_maps", [])}),
                sum(len(r.get("new_events", [])) for r in rows))

    window = [r for r in played if r["t"] > since]
    last = played[-1]
    before = [r for r in played if r["t"] <= since]
    steps_window = last["steps"] - (before[-1]["steps"] if before else played[0]["steps"])
    seconds_window = max(1.0, now - since)
    if steps_window > 0:
        acting = (f"{steps_window} steps in the last {seconds_window / 60:.0f} minutes: the harness is acting "
                  "on the game, so it is not frozen, whatever the trajectory and video show.")
    else:
        acting = (f"No steps in the last {seconds_window / 60:.0f} minutes: the game has received no action, so "
                  "the harness is waiting on a model call or is frozen. Its log says which.")
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
        f"As of {stamp(last['t'])} UTC, {max(0, now - last['t']):.0f} seconds before this was written (the server "
        f"records every 30 seconds of play, so later events are not in it yet): at {last.get('map')}, step "
        f"{last['steps']}, {badges} badge(s), {last.get('maps_reached')} maps "
        f"reached, {last.get('tiles')} grid spots stood on in all.",
        f"Steps taken: {acting}",
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
            "steps": last["steps"], "steps_window": steps_window, "tiles": last.get("tiles"), "maps_reached": last.get("maps_reached"),
            "badges": badges, "new_tiles_window": sum(r.get("new_tiles", 0) for r in window),
            "new_tiles_15": tiles_15, "new_tiles_30": tiles_30, "quiet_minutes": round(quiet_minutes, 1)}


FRAME_INDEX = "frame_index.jsonl"


def frame_locations(run):
    records = progress_records(run)
    if not records:
        return []
    begin, end = records[0]["t"] - 5, records[-1]["t"] + 3600
    found = []
    for session in sorted({r["session"] for r in records}):
        for index in glob.glob(os.path.join(OBS, session, "*", FRAME_INDEX)):
            base = os.path.dirname(index)
            for row in read_jsonl(index):
                first, last = row["frames"]
                if last > first and begin <= row["t"] <= end:
                    found.append({"t": row["t"], "step": row["step"], "path": os.path.join(base, row["video"]),
                                  "frame": last - 1, "approx": False})
    return sorted(found, key=lambda r: r["t"]) or approximate_locations(run, records)


def approximate_locations(run, records):
    import cv2
    times = [r["t"] for r in records]
    steps = [r["steps"] for r in records]

    def step_at(t):
        if t <= times[0]:
            return steps[0]
        for (t0, s0), (t1, s1) in zip(zip(times, steps), zip(times[1:], steps[1:])):
            if t0 <= t <= t1:
                return round(s0 + (s1 - s0) * (t - t0) / max(t1 - t0, 1e-6))
        return steps[-1]

    found = []
    for session in sorted({r["session"] for r in records}):
        opened = min(r["t"] for r in records if r["session"] == session)
        paths = sorted(glob.glob(os.path.join(OBS, session, "*", "videos", "*.mp4")), key=os.path.getmtime)
        previous = opened
        for path in paths:
            closed = os.path.getmtime(path)
            if closed < opened:
                continue
            capture = cv2.VideoCapture(path)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            for frame in range(count):
                t = previous + (frame + 1) * (closed - previous) / count
                found.append({"t": t, "step": step_at(t), "path": path, "frame": frame, "approx": True})
            previous = closed
    return found


def read_frames(locations):
    import cv2
    frames = []
    captures = {}
    for loc in locations:
        capture = captures.get(loc["path"]) or captures.setdefault(loc["path"], cv2.VideoCapture(loc["path"]))
        capture.set(cv2.CAP_PROP_POS_FRAMES, loc["frame"])
        ok, frame = capture.read()
        if ok:
            frames.append((loc, frame))
    for capture in captures.values():
        capture.release()
    return frames


def spread(items, count):
    if len(items) <= count:
        return list(items)
    return [items[round(i * (len(items) - 1) / max(1, count - 1))] for i in range(count)]


def frame_label(number, loc):
    when = datetime.fromtimestamp(loc["t"], timezone.utc).strftime("%H:%M:%S")
    return f"#{number} step {'~' if loc['approx'] else ''}{loc['step']} {when}"


def write_sheet(frames, path, columns=3):
    import cv2
    import numpy as np
    tiles = []
    for label, frame in frames:
        frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        frame = cv2.copyMakeBorder(frame, 22, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(frame, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(frame)
    while len(tiles) % columns:
        tiles.append(np.full_like(tiles[0], 255))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]))
    return path


def frame_sheets(run, since, until, out_dir, label, frames=12, per_sheet=6):
    picked = read_frames(spread([loc for loc in frame_locations(run) if since < loc["t"] <= until], frames))
    sheets = []
    for start in range(0, len(picked), per_sheet):
        chunk = [(frame_label(start + i + 1, loc), frame) for i, (loc, frame) in enumerate(picked[start:start + per_sheet])]
        sheets.append(write_sheet(chunk, os.path.join(out_dir, f"{label}_{start // per_sheet + 1}.png")))
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
        lines.append("No trajectory events in this window. The harness records a plan when it is made and an "
                     "executor leg when the leg ends, so an empty window means one long model call or one long leg "
                     "is under way. Whether the game is still being played is the 'Steps taken' line above; only "
                     "no steps at all means the harness has stopped.")
    for e in events:
        text = e["text"] if len(e["text"]) <= text_limit else e["text"][:text_limit] + " [...]"
        lines.append(f"- {stamp(e['t'])} episode {e['episode']} {e['layer']} {e['kind']} (step {e.get('steps')}): "
                     + text.replace("\n", "\n    "))
    lines += ["", "## What the screen showed", ""]
    sheets = frame_sheets(run, since, until, os.path.join(directory, "frames"), label) if directory else []
    if not sheets:
        lines.append("No frames in this window. The server closes a video after every minute of play, so this "
                     "happens when the window is under a minute old or the game was hardly played; the 'Steps "
                     "taken' line above says which.")
    else:
        lines.append("Frame sheets of this window, in time order (each frame labelled with its step and time): "
                     + ", ".join(sheets))
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
            if context.get("screen_review"):
                said = issue.get("screen_review")
                if not isinstance(said, str) or not said.startswith(SCREEN_STANCES):
                    errors.append(f"{name}.screen_review must start with one of {list(SCREEN_STANCES)}, then say "
                                  "what the screen review shows about this issue")
                else:
                    need_text(errors, said, f"{name}.screen_review")
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
    if context.get("screen_review"):
        said = draft.get("screen_review")
        need_text(errors, said, "screen_review (how this summary stands with the screen review)", 100)
        missing = [n for n in context.get("mismatches", [])
                   if not re.search(rf"\bepisode {n}\b", said or "", re.IGNORECASE)]
        if missing:
            errors.append("screen_review must take up each episode where the harness and the screen disagree, by "
                          f"number: say whether it is filed as an issue or why it is not; missing: episode(s) "
                          f"{missing}")
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


def crash(log_path):
    try:
        with open(log_path, "rb") as f:
            f.seek(max(0, os.path.getsize(log_path) - 20000))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    if "Traceback (most recent call last)" not in tail:
        return None
    after = tail.rsplit("Traceback (most recent call last)", 1)[1].splitlines()
    errors = [line.strip() for line in after if re.match(r"^[A-Za-z_.]*(Error|Exception)\b", line.strip())]
    return errors[-1] if errors else "a Python exception (see the log)"


def test_result(tid):
    info = read_json(os.path.join(TESTS, tid, "test.json")) or {}
    if info.get("crashed"):
        return "failed"
    verdicts = read_jsonl(os.path.join(TESTS, tid, "checks.jsonl"))
    last = verdicts[-1] if verdicts else {}
    if last.get("verdict") == "fixed" or (last.get("verdict") == "stop_point_reached" and last.get("passed") == "yes"):
        return "passed"
    if info.get("status") == "running":
        return "running"
    return "failed"


def investigation_path(context):
    return os.path.join(PLAYS, context["pid"], "investigations", f"{context['issue']['id']}.md")


SCREEN_STANCES = ("agrees", "disagrees", "not visible")
FIX_STATUSES = ("fixed", "deferred", "not_confirmed")
SECONDHAND = ("checks.md", "checks.jsonl", "issues.md", "issues.jsonl", "summary.md", "big_issues")


def validate_confirm(draft, context):
    errors = []
    if not isinstance(draft, dict):
        return ["the draft must be a JSON object"]
    issue = context["issue"]
    if draft.get("issue") != issue["id"]:
        errors.append(f"issue must be {issue['id']!r}, the one this step is about")
    if draft.get("confirmed") not in ("yes", "no"):
        errors.append("confirmed must be \"yes\" or \"no\"")
    need_text(errors, draft.get("failure"), "failure", 100)
    need_text(errors, draft.get("against"), "against")
    records = draft.get("records")
    if not isinstance(records, list):
        errors.append("records must be a list")
        records = []
    firsthand = 0
    for i, row in enumerate(records):
        if not isinstance(row, dict):
            errors.append(f"records[{i}] must be an object")
            continue
        for key in ("source", "where", "shows"):
            need_text(errors, row.get(key), f"records[{i}].{key}", 5 if key != "shows" else MIN_TEXT)
        if isinstance(row.get("source"), str) and not any(name in row["source"] for name in SECONDHAND):
            firsthand += 1
    if firsthand < 2:
        errors.append("records must cite at least 2 records made while the game was played (server progress "
                      "records, the harness trajectory or log, frames), not the checks' or conclusion's own account "
                      f"({', '.join(SECONDHAND)})")
    if draft.get("harness_at") != context["start_sha"] or draft.get("uncommitted"):
        errors.append("the confirmation comes before any change: it is only accepted while the harness is exactly "
                      "as it was when this step began, so revert or discard your changes and hand it in first")
    return errors


def validate_fix(draft, context):
    errors = []
    if not isinstance(draft, dict):
        return ["the draft must be a JSON object"]
    issue = context["issue"]
    if draft.get("issue") != issue["id"]:
        errors.append(f"issue must be {issue['id']!r}, the one this step is about")
    status = draft.get("status")
    if status not in FIX_STATUSES:
        errors.append(f"status must be one of {list(FIX_STATUSES)}")
    if "confirm" in context.get("accepts", []):
        confirmed = (read_json(os.path.join(STAGED, "confirm.json")) or {}).get("confirmed")
        if confirmed is None:
            errors.append("there is no accepted confirm record: confirm the failure in the records first")
        elif confirmed == "no" and status != "not_confirmed":
            errors.append("the confirmation found the failure is not in the records, so the status is not_confirmed")
        elif confirmed == "yes" and status == "not_confirmed":
            errors.append("the confirmation found the failure in the records, so the status is fixed or deferred")
    elif status == "not_confirmed":
        errors.append("not_confirmed is only for the deep dive of an issue")
    need_text(errors, draft.get("diagnosis"), "diagnosis", 100)
    path = investigation_path(context)
    try:
        with open(path) as f:
            written = f.read()
    except OSError:
        written = ""
    wanted = ["## Diagnosis", "## Evidence", "## Fix"]
    if "confirm" in context.get("accepts", []):
        wanted = ["## Confirmation"] + (wanted if status != "not_confirmed" else [])
    if issue.get("present_in_previous", "no") != "no" and status != "not_confirmed":
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
    if status in ("deferred", "not_confirmed"):
        need_text(errors, draft.get("note"), "note (why it is deferred, and what was tried)" if status == "deferred"
                  else "note (what the records show happened instead)")
        if git("diff", "--stat", context["start_sha"], "HEAD"):
            errors.append(f"a {status} issue leaves the harness as it was when this step began: revert your "
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
              "issues": validate_issues, "summary": validate_summary, "confirm": validate_confirm,
              "fix": validate_fix,
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
                           "screen_review": "agrees | disagrees | not visible: what the screen review shows "
                                            "about this issue",
                           "rank": 1}]},
    "summary": {"summary": "one account of the whole playthrough",
                "screen_review": "how this account stands with the screen review, taking up by number each "
                                 "episode where the harness and the screen disagree",
                "expectations": [{"held": "yes | no | unclear", "evidence": "what shows it"}]},
    "confirm": {"issue": "I1", "confirmed": "yes | no",
                "failure": "what the records show happened, at the place the issue names",
                "records": [{"source": "the record: a file path, or the digest.py / look.py command",
                             "where": "the time, step, episode or frame", "shows": "what it shows"}],
                "against": "the records that could contradict the issue as filed, and what they show"},
    "fix": {"issue": "I1", "status": "fixed | deferred | not_confirmed", "diagnosis": "the cause, as you found it",
            "commits": ["<sha>"], "note": "for deferred: why, and what was tried; for not_confirmed: what the "
                                          "records show happened instead"},
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
