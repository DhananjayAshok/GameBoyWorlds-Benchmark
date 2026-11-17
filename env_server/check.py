import json
import os
import secrets
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
                + check_test_policy(parameters=parameters))
    for failure in failures:
        log_info(f"FAIL {failure}", parameters=parameters)
    if not failures:
        log_info("env server policy checks passed", parameters=parameters)
    return not failures
