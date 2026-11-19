import json
import os
import secrets
import shutil
import socket
import tempfile
import threading

from env_server import codec, splits
from env_server.policy import Policy
from env_server.server import _handler
from env_server.sessions import SessionError, Sessions
from env_server.unix_http import unix_listener
from utils.log_handling import log_info


def _refused_reset(session, *, label: str) -> list:
    # Only the policy's own refusal counts: a crash must not pass for a policy.
    try:
        session.reset()
    except SessionError as exc:
        if exc.status != 403:
            return [f"{label} failed with {exc.status}, not the policy's 403: {exc}"]
        return []
    return [f"a test session allowed {label}"]


def check_test_policy(*, steps: int = 5, parameters: dict, action: str = "A") -> list:
    failures = []
    table = splits.load_splits(parameters=parameters)
    test_ids = [t for t, v in sorted(table.items()) if v["split"] == "test"]
    if not test_ids:
        return ["the split file has no test tasks"]

    sessions = Sessions(policy=Policy.of(kind="test"), max_sessions=1, idle_seconds=600,
                        record_dir=None, max_steps=None, parameters=parameters)
    if sessions.listed():
        failures.append("a test server listed tasks")
    if sessions.free_play():
        failures.append("a test server offered free-play start points")

    session = sessions.open(task_id=test_ids[0])
    payloads = [session.reset()]
    for _ in range(steps):
        payloads.append(session.step(action=action))
    payloads.append(session.observe())

    try:
        session.save()
        failures.append("a test session allowed save")
    except Exception:
        pass
    try:
        session.load(handle="whatever")
        failures.append("a test session allowed load")
    except Exception:
        pass
    failures += _refused_reset(session, label="a second reset")
    if session.describe().get("can_reset"):
        failures.append("a started test session says it can be reset")

    for i, payload in enumerate(payloads):
        extra = codec.audit_scored(payload=payload)
        if extra:
            failures.append(f"payload {i} carried tracker groups beyond the scored set: {extra}")
        state = payload.get("state")
        if state is not None and not set(state) <= set(codec.SCORED_GROUPS):
            failures.append(f"payload {i} state keys {sorted(state)} are not within {codec.SCORED_GROUPS}")

    # Task wordings repeat across games, so only a goal worded differently from this session's own is a leak.
    own_goal = table[test_ids[0]]["text"]
    other_goals = {v["text"] for t, v in table.items() if v["text"] and v["text"] != own_goal}
    blob = repr(payloads)
    for goal in other_goals:
        if len(goal) > 12 and goal in blob:
            failures.append(f"a response carried another task's goal text: {goal!r}")
            break

    sessions.close(session_id=session.id)

    session = sessions.open(task_id=test_ids[0])
    session.step(action=action)
    failures += _refused_reset(session, label="a reset after stepping without one")
    sessions.close(session_id=session.id)
    return failures


def check_dev_policy(*, parameters: dict) -> list:
    failures = []
    table = splits.load_splits(parameters=parameters)
    train_ids = [t for t, v in sorted(table.items()) if v["split"] == "train"]
    test_ids = [t for t, v in sorted(table.items()) if v["split"] in ("test", "test_env")]
    sessions = Sessions(policy=Policy.of(kind="dev"), max_sessions=1, idle_seconds=600,
                        record_dir=None, max_steps=None, parameters=parameters)
    for task_id in test_ids[:3]:
        try:
            sessions.open(task_id=task_id)
            failures.append(f"a dev server opened the held-back task {task_id}")
        except Exception:
            pass
    listed = {t["task_id"] for t in sessions.listed()}
    leaked = listed & set(test_ids)
    if leaked:
        failures.append(f"a dev server listed held-back tasks: {sorted(leaked)[:3]}")

    session = sessions.open(task_id=train_ids[0], want_state=True)
    payload = session.reset()
    if "state" not in payload:
        failures.append("a dev session that asked for state did not get it")
    else:
        served, reported = set(payload["state"]), set(session.env.get_info())
        missing = reported - served
        if missing:
            failures.append(f"a dev payload dropped tracker groups: {sorted(missing)}")
    try:
        session.reset()
    except SessionError as exc:
        failures.append(f"a dev session refused a second reset: {exc}")
    sessions.close(session_id=session.id)
    return failures


def check_playthrough_policy(*, steps: int = 6, parameters: dict, action: str = "button(a)") -> list:
    failures = []
    table = splits.load_splits(parameters=parameters)
    ids = [t for t, v in sorted(table.items()) if v["split"] == "playthrough"]
    if not ids:
        return ["the split file has no playthrough tasks"]
    other = next(t for t, v in sorted(table.items()) if v["split"] == "test")
    for kind in ("dev", "test"):
        if Policy.of(kind=kind).transitions:
            failures.append(f"a {kind} server serves obs.transition")
        sessions = Sessions(policy=Policy.of(kind=kind), max_sessions=1, idle_seconds=600,
                            record_dir=None, max_steps=None, parameters=parameters)
        try:
            sessions.open(task_id=ids[0])
            failures.append(f"a {kind} server opened the playthrough task {ids[0]}")
            sessions.close_all()
        except SessionError:
            pass

    directory = tempfile.mkdtemp(prefix="envcheck.", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
    video_dirs = []

    def serve() -> Sessions:
        return Sessions(policy=Policy.of(kind="playthrough"), max_sessions=4, idle_seconds=0,
                        record_dir=None, max_steps=None, parameters=parameters,
                        checkpoint_dir=directory, checkpoint_every=4)

    try:
        sessions = serve()
        if {t["task_id"] for t in sessions.listed()} != set(ids):
            failures.append("a playthrough server did not list exactly the playthrough tasks")
        if sessions.free_play():
            failures.append("a playthrough server offered free-play start points")
        try:
            sessions.open(task_id=other)
            failures.append(f"a playthrough server opened the test task {other}")
        except SessionError:
            pass

        session = sessions.open(task_id=ids[0])
        video_dirs.append(session.env._emulator.session_path)
        payloads = [session.reset()]
        for _ in range(steps):
            payloads.append(session.step(action=action))
        for i, payload in enumerate(payloads):
            extra = codec.audit_scored(payload=payload)
            if extra:
                failures.append(f"playthrough payload {i} carried tracker groups beyond the scored set: {extra}")
            for listed in ("button(", "wait("):
                if not any(text.startswith(listed) for text in payload["obs"]["actions"].values()):
                    failures.append(f"playthrough payload {i} does not list {listed}) in obs.actions")
            if ("transition" in payload["obs"]) != (i > 0):
                failures.append(f"playthrough payload {i} {'lacks' if i else 'carries'} obs.transition")
            elif i > 0 and not payload["obs"]["transition"]["frames"]:
                failures.append(f"playthrough step {i} served no transition frames")
        if not any(text.startswith("move(") for p in payloads for text in p["obs"]["actions"].values()):
            failures.append("a playthrough session never listed the state-wise move() action")
        for bad in ("A", "fly(somewhere)"):
            if session.step(action=bad)["valid"]:
                failures.append(f"a playthrough session accepted {bad!r}")
        if payloads[-1]["obs"]["max_steps"] != Policy.of(kind="playthrough").max_steps:
            failures.append(f"a playthrough session has a budget of {payloads[-1]['obs']['max_steps']} steps")
        failures += _refused_reset(session, label="a second reset")
        try:
            session.save()
            failures.append("a playthrough session allowed save")
        except SessionError:
            pass
        sessions.reap()
        try:
            sessions.get(session_id=session.id)
        except SessionError:
            failures.append("an idle playthrough session was reaped")
        if len(ids) > 1:
            try:
                sessions.open(task_id=ids[1])
                failures.append("a checkpointed run opened a second task")
            except SessionError:
                pass
        # Badges are matched on screen as they are won, so a restart can only keep them from the checkpoint.
        metric = session.env._emulator.state_tracker.metrics["subgoals"]
        metric._subgoals[0].completed = True
        before = session.observe()
        described = session.describe()
        token = session.token
        sessions.close_all()

        sessions = serve()
        session = sessions.open(task_id=ids[0])
        video_dirs.append(session.env._emulator.session_path)
        if (session.id, session.token) != (described["session"], token):
            failures.append("a restored session came back with a different id or token")
        after = session.describe()
        for field in ("steps", "subgoals", "episode_over", "success"):
            if after[field] != described[field]:
                failures.append(f"a restored session's {field} is {after[field]!r}, was {described[field]!r}")
        if session.observe()["obs"]["frame"] != before["obs"]["frame"]:
            failures.append("a restored session shows a different screen")
        failures += _refused_reset(session, label="a reset after a restart")
        if session.step(action=action)["obs"]["steps"] != described["steps"] + 1:
            failures.append("a restored session did not continue its step count")
        if session.step(action="wait()")["obs"]["steps"] != described["steps"] + 2:
            failures.append("wait() did not take exactly one step")

        marked = session.mark(label="episode7")
        marked_frame = session.observe()["obs"]["frame"]
        for _ in range(5):
            session.step(action="wait()")
        sessions.close_all()
        sessions = serve()
        session = sessions.open(task_id=ids[0])
        video_dirs.append(session.env._emulator.session_path)
        after = session.describe()
        if (after["steps"], after["label"], after["discarded_steps"]) != (marked["steps"], "episode7", 5):
            failures.append(f"a client checkpoint restored as steps={after['steps']}, label={after['label']!r}, "
                            f"discarded={after['discarded_steps']}; wanted {marked['steps']}, 'episode7', 5")
        if session.observe()["obs"]["frame"] != marked_frame:
            failures.append("a client checkpoint restored a different screen")
        sessions.close_all()
    finally:
        for path in [os.path.join(directory, name) for name in os.listdir(directory)]:
            os.unlink(path)
        os.rmdir(directory)
        for path in video_dirs:
            shutil.rmtree(path, ignore_errors=True)
            if os.path.isdir(os.path.dirname(path)) and not os.listdir(os.path.dirname(path)):
                os.rmdir(os.path.dirname(path))
    return failures


def _videos(directory: str) -> list:
    return sorted(os.path.join(root, name) for root, _, names in os.walk(directory)
                  for name in names if name.endswith(".mp4"))


def check_train_policy(*, steps: int = 4, parameters: dict, action: str = "button(a)") -> list:
    failures = []
    table = splits.load_splits(parameters=parameters)
    games = splits.train_sandbox_games(parameters=parameters)
    if not games:
        return ["split_config.yaml lists no train_sandbox_games"]
    allowed = {splits.playthrough_id(game=game) for game in games}
    playthroughs = {t for t, v in table.items() if v["split"] == "playthrough"}
    other = next(t for t, v in sorted(table.items()) if v["split"] == "test")
    root = tempfile.mkdtemp(prefix="envcheck.", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
    obs_dir, saves_dir = os.path.join(root, "obs"), os.path.join(root, "saves")
    progress_dir = os.path.join(root, "progress")

    def serve() -> Sessions:
        return Sessions(policy=Policy.of(kind="train"), max_sessions=1, idle_seconds=0, record_dir=None,
                        max_steps=None, parameters=parameters, obs_dir=obs_dir, saves_dir=saves_dir,
                        progress_dir=progress_dir)

    def progress_of(session_id: str) -> list:
        path = os.path.join(progress_dir, f"{session_id}.jsonl")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    try:
        try:
            Sessions(policy=Policy.of(kind="train"), max_sessions=1, idle_seconds=0, record_dir=None,
                     max_steps=None, parameters=parameters)
            failures.append("a train server started without a saves directory")
        except ValueError:
            pass
        sessions = serve()
        if {t["task_id"] for t in sessions.listed()} != allowed:
            failures.append(f"a train server listed {sorted(t['task_id'] for t in sessions.listed())}, "
                            f"not {sorted(allowed)}")
        for task_id in sorted(playthroughs - allowed) + [other]:
            try:
                sessions.open(task_id=task_id)
                failures.append(f"a train server opened {task_id}")
                sessions.close_all()
            except SessionError:
                pass

        task_id = sorted(allowed)[0]
        try:
            sessions.open(task_id=task_id, run="bad run/name")
            failures.append("a train server accepted the run name 'bad run/name'")
            sessions.close_all()
        except SessionError:
            pass
        session = sessions.open(task_id=task_id, want_state=True, run="check_run")
        first_id = session.id
        session.reset()
        for _ in range(steps):
            payload = session.step(action=action)
        if not codec.audit_scored(payload=payload):
            failures.append("a train session asked for state got only the scored groups")
        for bad in ("", "../escape", "a b", "x" * 129):
            try:
                session.save(name=bad)
                failures.append(f"a train session accepted the save name {bad!r}")
            except SessionError:
                pass
        saved = session.save(name="checkpointA")
        saved_frame = session.observe()["obs"]["frame"]
        for _ in range(steps):
            session.step(action="wait()")
        loaded = session.load(handle="checkpointA")
        if loaded["obs"]["frame"] != saved_frame or loaded["obs"]["steps"] != saved["steps"]:
            failures.append("loading a named save did not bring back its screen and step count")
        try:
            session.load(handle="neverSaved")
            failures.append("a train session loaded a save that does not exist")
        except SessionError as exc:
            if exc.status != 404:
                failures.append(f"loading a missing save failed with {exc.status}, not 404")
        session.reset()
        if session.reset()["obs"]["steps"] != 0:
            failures.append("a train session could not reset again")

        replacement = sessions.open(task_id=task_id)
        if not session.closed:
            failures.append("opening a second train session left the first open")
        sessions.close_all()

        sessions = serve()
        session = sessions.open(task_id=task_id)
        if replacement.id == session.id:
            failures.append("a new train server reused a session id")
        loaded = session.load(handle="checkpointA")
        if loaded["obs"]["frame"] != saved_frame or loaded["obs"]["steps"] != saved["steps"]:
            failures.append("a named save did not survive a server restart")
        session.step(action="wait()")
        sessions.close_all()

        first = progress_of(first_id)
        kinds = [r["kind"] for r in first]
        for kind in ("reset", "save", "load", "close"):
            if kind not in kinds:
                failures.append(f"a train session's progress log has no {kind!r} record (has {kinds})")
        if any(r["run"] != "check_run" for r in first):
            failures.append("a train session's progress records do not carry its run name")
        if first and not first[0]["map"]:
            failures.append("a train session's progress records name no map")
        reloaded = [r for r in progress_of(session.id) if r["kind"] == "load"]
        if not reloaded or reloaded[0]["maps_reached"] < 1:
            failures.append("a named save did not carry the maps reached across a server restart")
        if os.path.exists(os.path.join(progress_dir, "clears.jsonl")):
            failures.append("a train session that did not clear its game was recorded as a clear")

        videos = _videos(obs_dir)
        if len(videos) < 3:
            failures.append(f"a train server wrote {len(videos)} videos under obs_dir; wanted one per segment")
        if any(os.path.getsize(path) == 0 for path in videos):
            failures.append("a train server left an empty video")

        test_obs = os.path.join(root, "test_obs")
        test_id = next(t for t, v in sorted(table.items()) if v["split"] == "test")
        sessions = Sessions(policy=Policy.of(kind="test"), max_sessions=1, idle_seconds=600, record_dir=None,
                            max_steps=None, parameters=parameters, obs_dir=test_obs)
        session = sessions.open(task_id=test_id)
        session.reset()
        for _ in range(steps):
            session.step(action="A")
        sessions.close_all()
        if not _videos(test_obs):
            failures.append("a test server with obs_dir wrote no video")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return failures


def _raw_request(path: str, raw: bytes) -> tuple:
    # Raw bytes, so a check can send what a client library would refuse to: a malformed length, a non-ASCII bearer.
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(10)
        sock.connect(path)
        sock.sendall(raw)
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            data = b"".join(chunks)
            head, _, body = data.partition(b"\r\n\r\n")
            length = [int(l.split(b":")[1]) for l in head.split(b"\r\n") if l.lower().startswith(b"content-length:")]
            if length and len(body) >= length[0]:
                break
    data = b"".join(chunks)
    head, _, body = data.partition(b"\r\n\r\n")
    return int(head.split(b" ")[1]), body.decode("utf-8", "replace")


def check_http(*, parameters: dict) -> list:
    failures = []
    token = secrets.token_urlsafe(24)
    sessions = Sessions(policy=Policy.of(kind="dev"), max_sessions=1, idle_seconds=600,
                        record_dir=None, max_steps=None, parameters=parameters)
    directory = tempfile.mkdtemp(prefix="envcheck.", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
    path = os.path.join(directory, "check.sock")
    httpd = unix_listener(path, _handler(sessions=sessions, token=token, parameters=parameters))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        def get(bearer: bytes) -> tuple:
            auth = b"Authorization: Bearer " + bearer + b"\r\n" if bearer is not None else b""
            return _raw_request(path, b"GET /tasks HTTP/1.1\r\nHost: x\r\n" + auth + b"Connection: close\r\n\r\n")

        status, _ = get(token.encode())
        if status != 200:
            failures.append(f"the server token was refused ({status}); the remaining HTTP checks mean nothing")
        for label, bearer in (("no token", None), ("a wrong token of the right length", b"x" * len(token)),
                              ("the token's prefix", token[:-1].encode()),
                              ("a non-ASCII token", "é".encode("utf-8") * len(token))):
            status, body = get(bearer)
            if status != 401:
                failures.append(f"{label} got {status}, not 401: {body[:200]}")

        # An unanticipated failure must tell the client that it failed and nothing about why.
        status, body = _raw_request(path, b"POST /sessions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer "
                                     + token.encode() + b"\r\nContent-Length: not-a-number\r\nConnection: close\r\n\r\n")
        if status != 500:
            failures.append(f"a malformed request got {status}, not 500")
        try:
            reply = json.loads(body)
        except ValueError:
            reply = {}
        if set(reply) != {"error", "incident"} or reply.get("error") != "internal error":
            failures.append(f"a 500 carried more than an incident id: {body[:300]}")
        for tell in ("/home", "/project", "/scratch", "/opt", "Error", "Traceback", "line "):
            if tell in body:
                failures.append(f"a 500 body leaks {tell!r}: {body[:300]}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        os.unlink(path)
        os.rmdir(directory)
    return failures


def run(*, parameters: dict) -> bool:
    failures = (check_http(parameters=parameters) + check_dev_policy(parameters=parameters)
                + check_test_policy(parameters=parameters) + check_playthrough_policy(parameters=parameters)
                + check_train_policy(parameters=parameters))
    for failure in failures:
        log_info(f"FAIL {failure}", parameters=parameters)
    if not failures:
        log_info("env server policy checks passed", parameters=parameters)
    return not failures
