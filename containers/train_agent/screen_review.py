import argparse
import json
import os
import re

import cyclelib as lib

FRAMES_PER_EPISODE = 36
FRAMES_PER_SEGMENT = 6
OVERVIEW_FRAMES = 36
MAX_CALLS = 150
VERDICTS = ("yes", "no", "partly", "unclear")

SCREEN = ("The images are frames from a Game Boy game, in time order. Each frame is labelled with a number, the "
          "game step and the time (a step marked ~ is estimated). The blue pad in the corner shows the button "
          "being pressed in that frame, highlighted.")

EPISODE_QUESTION = """{screen}

These frames are one episode of an AI agent playing the game. You are told what the agent set out to do; you
are not told what it thinks happened, and you must judge only from the frames.

The episode's task: {task}

The agent broke the task into these subgoals. Each segment is one plan, in order (a new segment means the
agent planned again), and each frame's label starts with the number of its segment ("seg N").
{segments}

Reply with only a JSON object, no other text:
{{"task": {{"achieved": "yes | no | partly | unclear", "evidence": "the frame numbers that show it",
           "end_state": "where the player is and what is on screen at the end of the episode"}},
 "subgoals": [{{"segment": 1, "subgoal": "the subgoal, as given", "achieved": "yes | no | partly | unclear",
               "evidence": "the frame numbers", "instead": "what happened instead, when not yes"}}],
 "waste": "repeated, undone or pointless actions you can see, with frame numbers, or none",
 "notes": "anything that matters on screen that the goals do not mention, or none"}}
Judge every subgoal listed. "unclear" only when the frames cannot show it either way."""

OVERVIEW_QUESTION = """{screen}

These frames are a whole playthrough of the game by an AI agent, about one per minute of play. Below is, for
each episode, the task the agent set itself and what a reviewer judged from that episode's frames.

{episodes}

Reply with only a JSON object, no other text:
{{"progress": "how far the player got in the game, from the screen",
  "time": "where the playing time went: what took long, and what was quick",
  "patterns": ["a problem in how the player behaves that recurs across episodes, with the episodes"],
  "account": "one paragraph: the whole playthrough as it looked on screen"}}"""


class Budget(Exception):
    pass


def episodes_of(pid):
    events = lib.read_jsonl(os.path.join(lib.PLAYS, pid, "harness", "trajectory.jsonl"))
    episodes = {}
    for e in events:
        row = episodes.setdefault(e["episode"], {"episode": e["episode"], "task": None, "t": e["t"],
                                                 "step": e["steps"], "plans": [], "status": None})
        if e["layer"] == "strategist" and e["kind"] == "intention":
            row["task"] = e["text"].split("\n")[0].removeprefix("Task:").strip()
            row["t"], row["step"] = e["t"], e["steps"]
        elif e["layer"] == "strategist" and e["kind"] == "outcome":
            match = re.match(r"Status:\s*([\w-]+)", e["text"])
            row["status"] = match.group(1) if match else e["text"][:60]
        elif e["layer"] == "supervisor" and e["text"].startswith("[plan]") and "Subgoals:" in e["text"]:
            listed = e["text"].rsplit("Subgoals:", 1)[1]
            subgoals = [line.strip()[2:].strip() for line in listed.splitlines() if line.strip().startswith("- ")]
            if subgoals:
                row["plans"].append({"t": e["t"], "step": e["steps"], "subgoals": subgoals})
    return [episodes[k] for k in sorted(episodes) if episodes[k]["task"]]


def segments_of(episode, until_t):
    plans = episode["plans"] or [{"t": episode["t"], "step": episode["step"], "subgoals": []}]
    out = []
    for number, plan in enumerate(plans, start=1):
        start = episode["t"] if number == 1 else plan["t"]
        end = plans[number]["t"] if number < len(plans) else until_t
        out.append({"segment": number, "start": start, "end": end, "subgoals": plan["subgoals"]})
    return out


def parse(reply):
    text = reply.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


class Reviewer:
    def __init__(self, pid, ask=None):
        self.pid = pid
        self.dir = os.path.join(lib.PLAYS, pid)
        self.frames_dir = os.path.join(self.dir, "review")
        self.ask = ask or lib.look
        self.calls = 0

    def call(self, images, question):
        for attempt in range(2):
            if self.calls >= MAX_CALLS:
                raise Budget(f"the review stopped at its limit of {MAX_CALLS} model calls")
            self.calls += 1
            reply = self.ask(images, question if attempt == 0 else question + "\n\nReply with the JSON only.",
                             max_tokens=4000)
            try:
                return parse(reply)
            except ValueError:
                continue
        raise ValueError(f"the model did not reply with JSON: {reply[:300]}")

    def review_episode(self, episode, segments, locations):
        per_segment = max(2, min(FRAMES_PER_SEGMENT, FRAMES_PER_EPISODE // max(1, len(segments))))
        sheets, lines, number = [], [], 0
        for seg in segments:
            picked = lib.read_frames(lib.spread([l for l in locations if seg["start"] < l["t"] <= seg["end"]],
                                                per_segment))
            goals = "; ".join(seg["subgoals"]) or "(no subgoals were recorded for this stretch)"
            lines.append(f"- Segment {seg['segment']}: {goals}"
                         + ("" if picked else " (no frames recorded for this stretch)"))
            if not picked:
                continue
            labelled = []
            for loc, frame in picked:
                number += 1
                labelled.append((f"seg {seg['segment']} " + lib.frame_label(number, loc), frame))
            sheets.append(lib.write_sheet(labelled, os.path.join(
                self.frames_dir, f"episode_{episode['episode']:03d}_segment_{seg['segment']}.png")))
        if not sheets:
            return {"error": "no frames were recorded for this episode"}
        return self.call(sheets, EPISODE_QUESTION.format(screen=SCREEN, task=episode["task"],
                                                          segments="\n".join(lines)))

    def run(self):
        locations = lib.frame_locations(self.pid)
        episodes = episodes_of(self.pid)
        info = lib.read_json(os.path.join(self.dir, "playthrough.json")) or {}
        ended = info.get("ended") or (locations[-1]["t"] if locations else 0)
        rows = []
        for i, episode in enumerate(episodes):
            until = episodes[i + 1]["t"] if i + 1 < len(episodes) else ended
            segments = segments_of(episode, until)
            row = {"episode": episode["episode"], "task": episode["task"], "start": episode["t"], "end": until,
                   "step": episode["step"], "segments": segments, "harness_status": episode["status"]}
            try:
                row["screen"] = self.review_episode(episode, segments, locations)
            except Budget as error:
                row["screen"] = {"error": str(error)}
            except Exception as error:
                row["screen"] = {"error": f"{type(error).__name__}: {error}"}
            row["mismatch"] = mismatch(row)
            rows.append(row)
        lib.write_jsonl(os.path.join(self.dir, "screen_review.jsonl"), rows)
        overview = self.overview(rows, locations)
        lib.write_json(os.path.join(self.dir, "screen_review.json"),
                       {"overview": overview, "calls": self.calls, "approximate": any(l["approx"] for l in locations)})
        lib.write_text(os.path.join(self.dir, "screen_review.md"), render(self.pid, rows, overview, locations))
        return rows, overview

    def overview(self, rows, locations):
        picked = lib.read_frames(lib.spread(locations, OVERVIEW_FRAMES))
        if not picked:
            return {"error": "no frames were recorded for this playthrough"}
        sheets = []
        for start in range(0, len(picked), 12):
            chunk = [(lib.frame_label(start + i + 1, loc), frame) for i, (loc, frame) in enumerate(picked[start:start + 12])]
            sheets.append(lib.write_sheet(chunk, os.path.join(self.frames_dir, f"overview_{start // 12 + 1}.png"),
                                          columns=4))
        summary = "\n".join(
            f"- Episode {r['episode']} (step {r['step']}): task \"{r['task']}\"; on screen: "
            + (f"{r['screen']['task'].get('achieved')}; {r['screen']['task'].get('end_state')}"
               if "task" in r["screen"] else f"not reviewed ({r['screen'].get('error')})")
            for r in rows)
        try:
            return self.call(sheets, OVERVIEW_QUESTION.format(screen=SCREEN, episodes=summary))
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}


def mismatch(row):
    screen = row["screen"].get("task", {}).get("achieved") if "task" in row["screen"] else None
    claimed = row["harness_status"]
    if screen not in VERDICTS or screen == "unclear" or claimed is None:
        return None
    if claimed == "completed" and screen != "yes":
        return f"the harness reported the task completed; on screen it was {screen}"
    if claimed != "completed" and screen == "yes":
        return f"the harness reported {claimed}; on screen the task was achieved"
    return None


def render(pid, rows, overview, locations):
    stamp = lib.stamp
    lines = [f"# Screen review of {pid}", "",
             "Judged from the frames alone, given only the tasks and subgoals the harness set: the reviewer did not "
             "see the harness's outcomes, the executor's legs, the server's counters or the checks. The harness's "
             "own reported outcome is set beside it afterwards, and disagreements are flagged.", ""]
    if any(l["approx"] for l in locations):
        lines += ["This run has no frame index, so frame times and steps are estimated from when each video closed "
                  "(marked ~ on the sheets).", ""]
    lines += ["## The playthrough", ""]
    if "error" in overview:
        lines.append(f"No overall review: {overview['error']}")
    else:
        lines += [f"**Progress.** {overview.get('progress')}", "", f"**Time.** {overview.get('time')}", "",
                  "**Patterns.**"] + [f"- {p}" for p in overview.get("patterns") or ["none"]] + [
                  "", f"**Account.** {overview.get('account')}"]
    flagged = [r for r in rows if r["mismatch"]]
    lines += ["", "## Where the harness and the screen disagree", ""]
    lines += [f"- Episode {r['episode']} ({stamp(r['start'])}): {r['mismatch']}. Task: {r['task']}"
              for r in flagged] or ["- none"]
    lines += ["", "## Episodes", ""]
    for r in rows:
        s = r["screen"]
        lines.append(f"### Episode {r['episode']}: {stamp(r['start'])} to {stamp(r['end'])} UTC, from step {r['step']}")
        lines += ["", f"Task: {r['task']}", ""]
        if "error" in s:
            lines += [f"Not reviewed: {s['error']}", ""]
            continue
        task = s.get("task", {})
        lines += [f"- On screen: **{task.get('achieved')}** ({task.get('evidence')}). At the end: "
                  f"{task.get('end_state')}",
                  f"- The harness reported: {r['harness_status'] or 'no outcome'}"
                  + (f" (**disagrees**: {r['mismatch']})" if r["mismatch"] else "")]
        for g in s.get("subgoals") or []:
            lines.append(f"  - Segment {g.get('segment')}, \"{g.get('subgoal')}\": {g.get('achieved')} "
                         f"({g.get('evidence')})" + (f"; instead: {g.get('instead')}"
                                                     if g.get("achieved") != "yes" and g.get("instead") else ""))
        lines += [f"- Waste: {s.get('waste')}", f"- Notes: {s.get('notes')}", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Review a finished playthrough from the screen, given only the "
                                                 "tasks and subgoals its harness set.")
    parser.add_argument("pid")
    args = parser.parse_args()
    rows, overview = Reviewer(args.pid).run()
    print(f"reviewed {len(rows)} episodes of {args.pid}; "
          f"{sum(1 for r in rows if r['mismatch'])} disagree with the harness; see "
          f"{os.path.join(lib.PLAYS, args.pid, 'screen_review.md')}")


if __name__ == "__main__":
    main()
