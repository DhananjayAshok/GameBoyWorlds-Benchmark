import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

import cyclelib as lib

MODEL = os.environ["SANDBOX_MODEL"]
SERVER = os.environ["ENV_SERVER_URL"].rstrip("/")
TOKEN = os.environ.pop("ENV_SERVER_TOKEN")
GAMES = os.environ["TRAIN_GAMES"].split()
PROMPTS = os.path.join(lib.HERE, "prompts")
AGENT_LOGS = os.path.join(lib.WORK, "agent_logs")
RUN_LOGS = os.path.join(lib.WORK, "logs")
LOG = os.path.join(AGENT_LOGS, "cycle.log")
STOP_FILE = os.path.join(lib.WORK, "STOP")
LAUNCH_NOTE = os.path.join(AGENT_LOGS, "next_prompt_note.md")

STEP_SECONDS = {"start": 600, "check": 240, "conclude": 1800, "deepdive": 2700, "big_identify": 1200,
                "big_solve": 3600, "iteration": 900}
STEP_SECONDS = {step: int(os.environ.get(f"TRAIN_{step.upper()}_SECONDS", seconds))
                for step, seconds in STEP_SECONDS.items()}
ATTEMPTS = 3
MISSED_CHECKS_END = 2
QUICK_SECONDS = 20
QUICK_EXITS_STOP = 8
RETRY = ("Your record was not accepted yet:\n{errors}\n\nFix exactly these, in the same files, and hand the "
         "record in again with record.py. Do not start the step over.")


class Stop(Exception):
    def __init__(self, reason, status=0):
        super().__init__(reason)
        self.status = status


def log(message):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}"
    os.makedirs(AGENT_LOGS, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def server(method, path):
    request = urllib.request.Request(SERVER + path, method=method, headers={"Authorization": "Bearer " + TOKEN})
    try:
        with urllib.request.urlopen(request, timeout=60) as reply:
            return json.loads(reply.read())
    except urllib.error.HTTPError as error:
        return {"error": f"{error.code}"}
    except OSError as error:
        return {"error": str(error)}


def short(game):
    return "".join(c for c in game.replace("pokemon_", "") if c.isalpha())


def prompt(name, **fields):
    with open(os.path.join(PROMPTS, f"{name}.md")) as f:
        text = f.read()
    for key, value in fields.items():
        text = text.replace(f"[[{key}]]", str(value))
    return text


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    for sig, grace in ((signal.SIGINT, 20), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def launch(name, game, start, directory):
    os.makedirs(RUN_LOGS, exist_ok=True)
    command = [lib.PYTHON, "run_playthrough.py", "--remote", "--game", game, "--name", name, "--model", MODEL,
               "--vlm_kind", "gateway", "--max_episodes", "100000", "--record_dir", os.path.join(directory, "harness")]
    if start and start != lib.BEGINNING:
        command += ["--start_checkpoint", start]
    out = open(os.path.join(RUN_LOGS, f"{name}.log"), "ab")
    process = subprocess.Popen(command, cwd=lib.HARNESS, env=dict(os.environ, ENV_SERVER_TOKEN=TOKEN), stdout=out,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    out.close()
    return process, " ".join(command)


def commit_leftovers(why):
    if lib.dirty():
        lib.git("add", "-A")
        lib.git("commit", "-q", "-m", f"Uncommitted work {why}, committed by the controller")
        log(f"committed uncommitted harness work {why}")


def restore_harness(start_sha):
    lib.git("checkout", "--", ".", check=False)
    lib.git("clean", "-fdq", check=False)
    if not lib.git("diff", "--stat", start_sha, "HEAD"):
        return
    made = lib.commits_between(start_sha)
    out = subprocess.run(["git", "-C", lib.HARNESS, "revert", "--no-edit", *reversed(made)], capture_output=True)
    if out.returncode != 0:
        lib.git("revert", "--abort", check=False)
        lib.git("reset", "-q", "--hard", start_sha)
        log(f"reverting {len(made)} commit(s) did not apply; the harness was reset to {start_sha[:10]}")


def bullet(rows):
    return "\n".join(f"- {row}" for row in rows) or "- none"


class Controller:
    def __init__(self):
        for directory in (lib.CYCLE, lib.DRAFTS, lib.STAGED, lib.REQUESTS, lib.REPLIES, lib.PLAYS, lib.TESTS,
                          lib.ITERATIONS, AGENT_LOGS, RUN_LOGS):
            os.makedirs(directory, exist_ok=True)
        self.state = lib.read_json(lib.STATE) or {"cycle": 1, "phase": "start", "pid": None, "calls": 0}
        self.context = {}
        self.play = None
        self.test = None
        self.test_stop = False
        self.quick = 0

    def save(self):
        lib.write_json(lib.STATE, self.state)

    def set_context(self, **context):
        self.context = context
        lib.write_json(lib.CONTEXT, context)

    @property
    def pid(self):
        return self.state["pid"]

    @property
    def game(self):
        return GAMES[(self.state["cycle"] - 1) % len(GAMES)]

    def play_dir(self, *parts):
        return os.path.join(lib.PLAYS, self.pid, *parts)

    def check_stop(self):
        if os.path.exists(STOP_FILE):
            raise Stop("/work/STOP exists")

    def call_agent(self, text, resume, timeout, name):
        self.state["calls"] += 1
        self.save()
        command = ["prime-agent", "--offline", "--provider", "gateway", "--model", MODEL, "--cwd", lib.WORK]
        if resume:
            command.append("--continue")
        command += ["-p", text]
        path = os.path.join(AGENT_LOGS, f"{self.state['calls']:05d}_{name}.log")
        started = time.time()
        with open(path, "ab") as out:
            process = subprocess.Popen(command, cwd=lib.WORK, stdout=out, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            while process.poll() is None:
                time.sleep(1)
                self.tick()
                self.check_stop()
                if time.time() - started >= timeout:
                    log(f"{name}: the agent used its {int(timeout)}s; ending the call")
                    break
        finally:
            stop_process(process)
        elapsed = time.time() - started
        self.quick = self.quick + 1 if elapsed < QUICK_SECONDS else 0
        if self.quick >= QUICK_EXITS_STOP:
            raise Stop(f"{self.quick} agent calls in a row ended within {QUICK_SECONDS}s", status=1)

    def run_agent(self, step, text, kinds, attempts=ATTEMPTS):
        self.context["accepts"] = kinds
        self.context["step"] = step
        lib.write_json(lib.CONTEXT, self.context)
        for kind in kinds:
            for directory in (lib.DRAFTS, lib.STAGED):
                path = os.path.join(directory, f"{kind}.json")
                if os.path.exists(path):
                    os.remove(path)
        deadline = time.time() + STEP_SECONDS[step]
        errors = []
        staged = {}
        for attempt in range(attempts):
            remaining = deadline - time.time()
            if remaining < 30:
                break
            message = text if attempt == 0 else RETRY.format(errors=bullet(errors))
            self.call_agent(message, attempt > 0, remaining, f"{step}_{attempt + 1}")
            staged = {kind: lib.read_json(os.path.join(lib.STAGED, f"{kind}.json")) for kind in kinds}
            errors = []
            for kind in kinds:
                if staged[kind] is None:
                    errors.append(f"no accepted {kind} record: write it and hand it in with record.py {kind}")
                else:
                    errors += [f"{kind}: {e}" for e in lib.VALIDATORS[kind](staged[kind], self.context)]
            if not errors:
                return staged, True
        log(f"{step}: no accepted record after {attempts} attempt(s): {errors}")
        valid = {kind: draft for kind, draft in staged.items()
                 if draft is not None and not lib.VALIDATORS[kind](draft, self.context)}
        return valid, False

    def tick(self):
        for name in sorted(os.listdir(lib.REQUESTS)):
            path = os.path.join(lib.REQUESTS, name)
            request = lib.read_json(path)
            os.remove(path)
            if not request:
                continue
            reply = self.start_test(request) if request.get("kind") == "start_test" else self.stop_test()
            lib.write_json(os.path.join(lib.REPLIES, f"{request['id']}.json"), reply)
        if self.test is not None:
            info = self.test["info"]
            if self.test_stop:
                self.end_test("a verdict ended it")
            elif self.test["process"].poll() is not None:
                self.end_test("the run ended by itself; see its log")
            elif time.time() - info["started"] >= info["minutes"] * 60:
                self.end_test(f"the {info['minutes']}-minute limit")

    def stop_test(self):
        self.test_stop = True
        return {"ok": True}

    def start_test(self, request):
        issue = self.context.get("issue")
        if self.context.get("step") not in ("deepdive", "big_solve") or issue is None:
            return {"ok": False, "error": "short tests can only be run while you are working on an issue"}
        tests = self.context.setdefault("tests", [])
        if self.test is not None:
            return {"ok": False, "error": f"short test {self.test['info']['tid']} is still running"}
        if len(tests) >= lib.TESTS_PER_ISSUE:
            return {"ok": False, "error": f"this issue has used its {lib.TESTS_PER_ISSUE} short tests"}
        if lib.dirty():
            return {"ok": False, "error": "the harness has uncommitted changes; commit the fix before testing it"}
        for part in ("test", "stop"):
            if len(str(request.get(part) or "").strip()) < 20:
                return {"ok": False, "error": f"--{part} must say concretely what to look for"}
        try:
            minutes = float(request.get("minutes") or lib.SHORT_TEST_MINUTES)
        except (TypeError, ValueError):
            return {"ok": False, "error": "--minutes must be a number"}
        if not 0 < minutes <= lib.SHORT_TEST_MINUTES:
            return {"ok": False, "error": f"a short test runs at most {lib.SHORT_TEST_MINUTES} minutes"}
        start = request.get("start")
        if start == lib.BEGINNING:
            game = request.get("game")
            if game not in GAMES:
                return {"ok": False, "error": f"--from {lib.BEGINNING} needs --game, one of {GAMES}"}
        else:
            checkpoint, problem = self.usable_checkpoint(start)
            if problem:
                return {"ok": False, "error": problem}
            game = checkpoint["game"]
        tid = f"t{self.state['cycle']:03d}_{issue['id'].lower()}_{chr(97 + len(tests))}_{short(game)}"
        while os.path.exists(os.path.join(lib.TESTS, tid)):
            tid = tid.replace(f"_{short(game)}", f"x_{short(game)}")
        directory = os.path.join(lib.TESTS, tid)
        os.makedirs(directory)
        info = {"tid": tid, "cycle": self.state["cycle"], "pid": self.pid, "issue": issue["id"], "game": game,
                "from": start, "test": request["test"], "stop": request["stop"], "minutes": minutes,
                "started": time.time(), "commit": lib.head(), "status": "running"}
        lib.write_json(os.path.join(directory, "test.json"), info)
        process, info["command"] = launch(tid, game, start, directory)
        lib.write_json(os.path.join(directory, "test.json"), info)
        self.test = {"info": info, "process": process, "directory": directory}
        self.test_stop = False
        tests.append(tid)
        lib.write_json(lib.CONTEXT, self.context)
        log(f"short test {tid} started from {start} for issue {issue['id']}")
        return {"ok": True, "tid": tid, "commit": info["commit"], "minutes": minutes}

    def end_test(self, reason):
        stop_process(self.test["process"])
        info = dict(self.test["info"], status="ended", end_reason=reason, ended=time.time())
        lib.write_json(os.path.join(self.test["directory"], "test.json"), info)
        log(f"short test {info['tid']} ended: {reason}")
        self.test = None
        self.test_stop = False

    def usable_checkpoint(self, ref):
        checkpoint, problem = lib.find_checkpoint(ref)
        if problem:
            return None, problem
        saves = server("GET", "/saves").get("saves")
        if saves is not None and not any(s["game"] == checkpoint["game"] and s["name"] == checkpoint["server_save"]
                                         for s in saves):
            return None, f"the server no longer has the game state of checkpoint {ref}"
        return checkpoint, None

    def startable(self, game):
        rows = []
        for other in sorted(os.listdir(lib.PLAYS)):
            info = lib.read_json(os.path.join(lib.PLAYS, other, "playthrough.json")) or {}
            if info.get("game") != game or other == self.pid:
                continue
            final = info.get("final_checkpoint")
            if final and lib.find_checkpoint(final)[1] is None:
                rows.append(f"{final}: the end of playthrough {other} ({info.get('end_reason')}, after "
                            f"{info.get('minutes')} minutes)")
            for kept in lib.read_jsonl(os.path.join(lib.PLAYS, other, "save_states.jsonl")):
                if lib.find_checkpoint(kept["id"])[1] is None:
                    rows.append(f"{kept['id']}: {kept['context']} Useful for: {kept['useful_for']}")
        return rows

    def expectations(self, game):
        for cycle in range(self.state["cycle"] - 1, 0, -1):
            iteration = lib.read_json(os.path.join(lib.ITERATIONS, f"{cycle:03d}.json"))
            if iteration:
                return [row["text"] for row in iteration.get("expectations", []) if row["game"] in (game, "both")]
        return []

    def previous_pid(self, game=None):
        earlier = []
        for other in sorted(os.listdir(lib.PLAYS)):
            info = lib.read_json(os.path.join(lib.PLAYS, other, "playthrough.json")) or {}
            if other != self.pid and (game is None or info.get("game") == game):
                earlier.append(other)
        return earlier[-1] if earlier else None

    def phase_start(self):
        game = self.game
        self.state["pid"] = f"p{self.state['cycle']:03d}_{short(game)}"
        self.save()
        commit_leftovers("at the start of a playthrough")
        self.set_context(pid=self.pid, game=game)
        note = ""
        if os.path.exists(LAUNCH_NOTE):
            with open(LAUNCH_NOTE) as f:
                note = f"A note from the launcher. Do what it says first:\n\n{f.read().strip()}\n\n"
            os.remove(LAUNCH_NOTE)
        staged, _ = self.run_agent("start", prompt(
            "start", PID=self.pid, GAME=game, CYCLE=self.state["cycle"], NOTE=note,
            CHECKPOINTS=bullet(self.startable(game)),
            LAST_ITERATION=os.path.join(lib.ITERATIONS, f"{self.state['cycle'] - 1:03d}.md")), ["start"])
        choice = staged.get("start") or {"from": lib.BEGINNING, "why": "No accepted choice; started from the beginning."}
        if choice["from"] != lib.BEGINNING and self.usable_checkpoint(choice["from"])[1]:
            log(f"start: {choice['from']} cannot be used; starting from the beginning")
            choice = {"from": lib.BEGINNING, "why": f"{choice['from']} could not be loaded; started from the beginning."}
        previous = self.previous_pid(game)
        earlier = (lib.read_json(os.path.join(lib.PLAYS, previous, "playthrough.json")) or {}) if previous else {}
        since = lib.git("log", "--oneline", f"{earlier['harness_commit']}..HEAD").splitlines() \
            if earlier.get("harness_commit") else []
        os.makedirs(self.play_dir(), exist_ok=True)
        process, command = launch(self.pid, game, choice["from"], self.play_dir())
        info = {"pid": self.pid, "cycle": self.state["cycle"], "game": game, "from": choice["from"],
                "why": choice["why"], "harness_commit": lib.head(),
                "harness_subject": lib.git("log", "-1", "--format=%s"),
                "commits_since_previous_playthrough_of_this_game": since, "model": MODEL, "command": command,
                "started": time.time(), "started_utc": lib.stamp(time.time())}
        lib.write_json(self.play_dir("playthrough.json"), info)
        self.play = process
        self.state.update(phase="long", played=0, segment_started=time.time(), check=0, missed=0,
                          last_check=time.time())
        self.save()
        log(f"playthrough {self.pid} started from {choice['from']} on harness {info['harness_commit'][:10]}")

    def played(self):
        return self.state["played"] + time.time() - self.state["segment_started"]

    def phase_long(self):
        if self.play is None:
            self.play, _ = launch(self.pid, self.game, None, self.play_dir())
            played = self.state["played"] + max(0.0, self.state["last_check"] - self.state["segment_started"])
            self.state.update(played=played, segment_started=time.time(), last_check=time.time())
            self.save()
            log(f"playthrough {self.pid} relaunched after a controller restart")
        reason = None
        while reason is None:
            time.sleep(2)
            self.check_stop()
            if self.game in lib.cleared_games() and any(
                    r.get("kind") == "cleared" for r in lib.progress_records(self.pid)):
                reason = "the game was cleared"
            elif self.play.poll() is not None:
                reason = "the run ended by itself; see its log"
            elif self.played() >= lib.LONG_SECONDS:
                reason = f"the {lib.LONG_SECONDS // 60}-minute limit"
            elif time.time() - self.state["last_check"] >= lib.CHECK_SECONDS:
                verdict = self.do_check()
                if verdict == "stuck":
                    reason = "the check judged it stuck"
                elif self.state["missed"] >= MISSED_CHECKS_END:
                    reason = f"{self.state['missed']} checks in a row were not completed"
        stop_process(self.play)
        self.play = None
        points = lib.checkpoints(self.pid)
        info = lib.read_json(self.play_dir("playthrough.json"))
        info.update(ended=time.time(), ended_utc=lib.stamp(time.time()), end_reason=reason,
                    minutes=round(self.played() / 60, 1), checks=self.state["check"],
                    final_checkpoint=f"{self.pid}:{points[-1]['id']}" if points else None)
        lib.write_json(self.play_dir("playthrough.json"), info)
        self.state.update(phase="conclude")
        self.save()
        log(f"playthrough {self.pid} ended after {info['minutes']} minutes: {reason}")

    def do_check(self):
        now = time.time()
        since = self.state["last_check"]
        number = self.state["check"] + 1
        self.state.update(check=number, last_check=now)
        self.save()
        points = lib.checkpoints(self.pid)
        checkpoint = dict(points[-1], ref=f"{self.pid}:{points[-1]['id']}") if points else None
        measured = lib.signal(self.pid, since, now)
        digest_path = self.play_dir("digests", f"check_{number:03d}.md")
        try:
            lib.write_text(digest_path, lib.digest(self.pid, since, now, f"check_{number:03d}"))
        except Exception as error:
            lib.write_text(digest_path, f"The digest could not be built: {type(error).__name__}: {error}\n")
            log(f"check {number}: digest failed: {type(error).__name__}: {error}")
        expectations = self.expectations(self.game)
        self.set_context(pid=self.pid, game=self.game, check=number, recommendation=measured["recommendation"])
        staged, ok = self.run_agent("check", prompt(
            "check", PID=self.pid, GAME=self.game, NUMBER=number, MINUTES=round(self.played() / 60),
            DIGEST=digest_path, CHECKS=self.play_dir("checks.md"), SIGNAL=measured["text"],
            EXPECTATIONS=bullet(expectations), LOG=os.path.join(RUN_LOGS, f"{self.pid}.log")), ["check"], attempts=2)
        entry = {"check": number, "t": now, "utc": lib.stamp(now), "minutes": round(self.played() / 60, 1),
                 "checkpoint": checkpoint, "recommendation": measured["recommendation"],
                 "signal": {k: v for k, v in measured.items() if k != "text"}, "signal_text": measured["text"],
                 "digest": digest_path}
        if ok:
            entry.update(staged["check"])
            self.state["missed"] = 0
        else:
            entry["missed"] = True
            self.state["missed"] += 1
        self.save()
        lib.append_jsonl(self.play_dir("checks.jsonl"), entry)
        self.render_checks()
        verdict = entry.get("stuck", {}).get("verdict")
        log(f"check {number} of {self.pid}: {'missed' if not ok else verdict}; controller recommended "
            f"{measured['recommendation']}")
        return verdict

    def render_checks(self):
        lines = [f"# Checks of {self.pid}", ""]
        for row in lib.read_jsonl(self.play_dir("checks.jsonl")):
            checkpoint = (row.get("checkpoint") or {}).get("ref")
            lines += [f"## Check {row['check']}: {row['utc']} UTC, minute {row['minutes']}", "",
                      f"Latest checkpoint: {checkpoint}", "", row["signal_text"], ""]
            if row.get("missed"):
                lines += ["The check was not completed.", ""]
                continue
            lines += [f"**Intentions.** {row['summary']['intentions']}", "",
                      f"**Plans.** {row['summary']['plans']}", "",
                      f"**Attempts.** {row['summary']['attempts']}", "",
                      f"**On screen.** {row['summary']['on_screen']}", ""]
            for issue in row["issues"]:
                lines += [f"- Issue `{issue['code']}` at `{issue['locus']}`: {issue['observation']} "
                          f"Cause: {issue['cause']} Correct behaviour: {issue['correct_behavior']}"]
            stuck = row["stuck"]
            lines += ["", f"**Verdict: {stuck['verdict']}.** {stuck['reasoning']}"
                      + (f" Disagreement with the controller: {stuck['disagreement']}"
                         if stuck.get("disagreement") else ""), ""]
        lib.write_text(self.play_dir("checks.md"), "\n".join(lines) + "\n")

    def render_issues(self, name, title):
        rows = lib.read_jsonl(self.play_dir(f"{name}.jsonl"))
        lines = [f"# {title} of {self.pid}", ""]
        for row in rows:
            fix = row.get("fix")
            lines += [f"## {row['id']} (rank {row['rank']})"
                      + (f": `{row['code']}` at `{row['locus']}`" if "code" in row else ""), "",
                      row["description"], "", f"- Where: {row['where']}",
                      f"- Probable cause: {row['probable_cause']}",
                      f"- Present in previous: {row['present_in_previous']}"]
            if "likely_gain" in row:
                lines.append(f"- Likely gain: {row['likely_gain']}; solvability: {row['solvability']}")
            if "proposed_change" in row:
                lines.append(f"- Proposed change: {row['proposed_change']}")
            if fix is None:
                lines.append("- Fix: (not addressed)" if row.get("skipped") else "- Fix:")
            else:
                lines.append(f"- Fix: {fix['status']}; commits {[c[:10] for c in fix['commits']] or 'none'}; "
                             f"short tests {fix['tests'] or 'none'}. {fix.get('note') or fix.get('diagnosis')}")
            lines.append("")
        lib.write_text(self.play_dir(f"{name}.md"), "\n".join(lines) + "\n")

    def phase_conclude(self):
        expectations = self.expectations(self.game)
        previous = self.previous_pid()
        self.set_context(pid=self.pid, game=self.game, expectations=expectations)
        staged, _ = self.run_agent("conclude", prompt(
            "conclude", PID=self.pid, GAME=self.game, DIR=self.play_dir(),
            LOG=os.path.join(RUN_LOGS, f"{self.pid}.log"), LIMIT=lib.keep_limit(self.pid),
            CANDIDATES=bullet(lib.candidates(self.pid)), EXPECTATIONS=bullet(expectations),
            PREVIOUS=os.path.join(lib.PLAYS, previous, "issues.md") if previous else "(there is none yet)",
            TOP=lib.ISSUES_PER_CYCLE), ["saves", "issues", "summary"])
        points = {f"{self.pid}:{row['id']}": row for row in lib.checkpoints(self.pid)}
        lib.write_jsonl(self.play_dir("save_states.jsonl"),
                        [dict(row, server_save=points[row["id"]]["server_save"])
                         for row in (staged.get("saves") or {"keep": []})["keep"]])
        issues = sorted((staged.get("issues") or {"issues": []})["issues"], key=lambda row: row["rank"])
        for row in issues:
            row["fix"] = None
            row["skipped"] = row["rank"] > lib.ISSUES_PER_CYCLE
        lib.write_jsonl(self.play_dir("issues.jsonl"), issues)
        self.render_issues("issues", "Issues")
        summary = staged.get("summary") or {"summary": "The conclusion step did not produce a summary.",
                                            "expectations": []}
        lines = [f"# Playthrough {self.pid}", "", summary["summary"], ""]
        for text, row in zip(expectations, summary["expectations"]):
            lines.append(f"- Expected: {text} Held: {row['held']}. {row['evidence']}")
        lib.write_text(self.play_dir("summary.md"), "\n".join(lines) + "\n")
        self.state.update(phase="deepdive")
        self.save()
        log(f"conclusion of {self.pid}: {len(issues)} issue(s), "
            f"{len((staged.get('saves') or {'keep': []})['keep'])} checkpoint(s) kept")

    def solve(self, name, step, row):
        commit_leftovers("before an issue step")
        start_sha = lib.head()
        self.set_context(pid=self.pid, game=self.game, issue=row, start_sha=start_sha, tests=[])
        previous = row.get("present_in_previous", "no")
        earlier = ""
        if previous != "no":
            other, _, issue_id = previous.partition(":")
            earlier = prompt("earlier_fix", PREVIOUS=previous,
                             INVESTIGATION=os.path.join(lib.PLAYS, other, "investigations", f"{issue_id}.md"),
                             ISSUES=os.path.join(lib.PLAYS, other, "issues.md"))
        staged, ok = self.run_agent(step, prompt(
            step, PID=self.pid, GAME=self.game, DIR=self.play_dir(), ISSUE=json.dumps(row, indent=1),
            ISSUE_ID=row["id"], INVESTIGATION=lib.investigation_path(self.context), EARLIER=earlier,
            START=start_sha[:10], TESTS=lib.TESTS_PER_ISSUE, MINUTES=lib.SHORT_TEST_MINUTES,
            CHECKPOINTS=bullet(f"{self.pid}:{p['id']} (episode {p['episode']}, step {p['steps']}, "
                               f"{lib.stamp(p['t'])} UTC)" for p in lib.checkpoints(self.pid)[-60:])), ["fix"])
        if self.test is not None:
            self.end_test("the issue step ended")
        tests = [f"{tid}: {lib.test_result(tid)}" for tid in self.context.get("tests", [])]
        if ok:
            draft = staged["fix"]
            fix = {"status": draft["status"], "commits": lib.commits_between(start_sha)
                   if draft["status"] == "fixed" else [], "tests": tests, "diagnosis": draft["diagnosis"],
                   "note": draft.get("note"), "investigation": lib.investigation_path(self.context)}
        else:
            restore_harness(start_sha)
            fix = {"status": "deferred", "commits": [], "tests": tests, "diagnosis": None,
                   "note": "The step ended without an accepted record, so the harness was put back as it was.",
                   "investigation": lib.investigation_path(self.context)}
        rows = lib.read_jsonl(self.play_dir(f"{name}.jsonl"))
        for other in rows:
            if other["id"] == row["id"]:
                other["fix"] = fix
        lib.write_jsonl(self.play_dir(f"{name}.jsonl"), rows)
        self.render_issues(name, "Issues" if name == "issues" else "Larger issues")
        log(f"{row['id']} of {self.pid}: {fix['status']}, commits {[c[:10] for c in fix['commits']]}, tests {tests}")

    def solve_all(self, name, step, limit):
        while True:
            rows = [row for row in lib.read_jsonl(self.play_dir(f"{name}.jsonl"))
                    if row.get("fix") is None and row["rank"] <= limit]
            if not rows:
                return
            self.solve(name, step, rows[0])

    def phase_deepdive(self):
        self.solve_all("issues", "deepdive", lib.ISSUES_PER_CYCLE)
        self.state.update(phase="big_identify")
        self.save()

    def phase_big_identify(self):
        self.set_context(pid=self.pid, game=self.game)
        previous = self.previous_pid()
        staged, _ = self.run_agent("big_identify", prompt(
            "big_identify", PID=self.pid, GAME=self.game, DIR=self.play_dir(), MAX=lib.BIG_ISSUES_PER_CYCLE,
            PREVIOUS=os.path.join(lib.PLAYS, previous) if previous else "(there is none yet)"), ["big_issues"])
        issues = sorted((staged.get("big_issues") or {"issues": []})["issues"], key=lambda row: row["rank"])
        for row in issues:
            row["fix"] = None
        lib.write_jsonl(self.play_dir("big_issues.jsonl"), issues)
        self.render_issues("big_issues", "Larger issues")
        self.state.update(phase="big_solve")
        self.save()
        log(f"larger issues of {self.pid}: {len(issues)}")

    def phase_big_solve(self):
        self.solve_all("big_issues", "big_solve", lib.BIG_ISSUES_PER_CYCLE)
        self.state.update(phase="iteration")
        self.save()

    def phase_iteration(self):
        rows = lib.read_jsonl(self.play_dir("issues.jsonl")) + lib.read_jsonl(self.play_dir("big_issues.jsonl"))
        fixed = [{"issue": row["id"], "commits": row["fix"]["commits"]} for row in rows
                 if row.get("fix") and row["fix"]["status"] == "fixed"]
        deferred = [row["id"] for row in rows if row.get("fix") and row["fix"]["status"] == "deferred"]
        self.set_context(pid=self.pid, game=self.game, fixed=fixed, deferred=deferred, games=GAMES)
        cycle = self.state["cycle"]
        staged, ok = self.run_agent("iteration", prompt(
            "iteration", PID=self.pid, GAME=self.game, DIR=self.play_dir(), CYCLE=cycle,
            NEXT_GAME=GAMES[cycle % len(GAMES)], FIXED=bullet(f"{r['issue']}: {[c[:10] for c in r['commits']]}"
                                                              for r in fixed),
            DEFERRED=bullet(deferred), TESTS=lib.TESTS), ["iteration"])
        draft = staged.get("iteration") or {
            "summary": "The iteration step did not produce a summary.", "expectations": [],
            "changes": [{"commit": c, "issue": r["issue"], "what": ""} for r in fixed for c in r["commits"]],
            "deferred": [{"issue": issue, "why": ""} for issue in deferred]}
        lib.write_json(os.path.join(lib.ITERATIONS, f"{cycle:03d}.json"), dict(draft, pid=self.pid, cycle=cycle))
        lines = [f"# Iteration {cycle} (after playthrough {self.pid})", "", draft["summary"], "", "## Changes", ""]
        lines += [f"- `{str(row.get('commit'))[:10]}` ({row.get('issue')}): {row.get('what')}"
                  for row in draft["changes"]] or ["- none"]
        lines += ["", "## Deferred", ""]
        lines += [f"- {row.get('issue')}: {row.get('why')}" for row in draft["deferred"]] or ["- none"]
        lines += ["", "## What the next playthroughs should show", ""]
        lines += [f"- ({row['game']}) {row['text']}" for row in draft["expectations"]] or ["- none"]
        lib.write_text(os.path.join(lib.ITERATIONS, f"{cycle:03d}.md"), "\n".join(lines) + "\n")
        self.delete_checkpoints()
        log(f"iteration {cycle} written; {len(fixed)} fixed, {len(deferred)} deferred")
        self.state.update(cycle=cycle + 1, phase="start", pid=None)
        self.save()

    def delete_checkpoints(self):
        info = lib.read_json(self.play_dir("playthrough.json")) or {}
        keep = {row["id"] for row in lib.read_jsonl(self.play_dir("save_states.jsonl"))}
        if info.get("final_checkpoint"):
            keep.add(info["final_checkpoint"])
        runs = [(self.pid, self.game)]
        for tid in sorted(os.listdir(lib.TESTS)):
            test = lib.read_json(os.path.join(lib.TESTS, tid, "test.json")) or {}
            if test.get("cycle") == self.state["cycle"]:
                runs.append((tid, test["game"]))
        points = [(run, game, row) for run, game in runs for row in lib.checkpoints(run)]
        kept_memory = {path for run, _, row in points if f"{run}:{row['id']}" in keep for path in row["memory"]}
        gone = lib.deleted_saves()
        removed = 0
        for run, game, row in points:
            if f"{run}:{row['id']}" in keep or row["server_save"] in gone:
                continue
            server("DELETE", f"/saves/{game}/{row['server_save']}")
            lib.append_jsonl(lib.DELETED, {"t": time.time(), "run": run, "id": row["id"],
                                           "server_save": row["server_save"]})
            gone.add(row["server_save"])
            removed += 1
            for path in row["memory"]:
                if path not in kept_memory and os.path.exists(path):
                    os.remove(path)
        log(f"checkpoints of {self.pid} and its short tests: kept {sorted(keep)}, deleted {removed}")

    def loop(self):
        while True:
            self.check_stop()
            if GAMES and all(game in lib.cleared_games() for game in GAMES):
                raise Stop(f"every game is cleared: {GAMES}")
            getattr(self, f"phase_{self.state['phase']}")()


def main():
    controller = Controller()
    log(f"controller started at cycle {controller.state['cycle']}, phase {controller.state['phase']}")
    try:
        controller.loop()
    except Stop as stop:
        log(f"stopping: {stop}")
        status = stop.status
    finally:
        stop_process(controller.play)
        if controller.test is not None:
            stop_process(controller.test["process"])
    sys.exit(status)


if __name__ == "__main__":
    main()
