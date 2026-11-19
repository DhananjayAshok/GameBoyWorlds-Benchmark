import hmac
import json
import os
import signal
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from env_server.sessions import SessionError, Sessions
from env_server.unix_http import unix_listener
from utils.log_handling import log_info, log_warn

REAP_EVERY_SECONDS = 60


def _same_token(given: Optional[str], expected: Optional[str]) -> bool:
    # Compared as bytes: hmac.compare_digest raises on a non-ASCII str.
    if not given or not expected:
        return False
    return hmac.compare_digest(given.encode("utf-8", "surrogatepass"), expected.encode("utf-8", "surrogatepass"))


def _handler(*, sessions: Sessions, token: str, parameters: dict):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "gameboy-env-server"

        def log_message(self, fmt, *args):
            pass

        def _send(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _bearer(self) -> Optional[str]:
            header = self.headers.get("Authorization", "")
            return header[7:].strip() if header.startswith("Bearer ") else None

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                raise SessionError("body is not JSON")

        def _parts(self) -> list:
            return [p for p in self.path.split("?")[0].strip("/").split("/") if p]

        def _authorize(self, *, parts: list) -> None:
            # The server token reaches everything; a session token reaches only its own session.
            given = self._bearer()
            session_id = parts[1] if len(parts) >= 2 and parts[0] == "sessions" else None
            if _same_token(given, token):
                return
            if session_id is not None and _same_token(given, sessions.token_of(session_id=session_id)):
                return
            raise SessionError("unauthorized", 401)

        def _route(self, method: str) -> dict:
            parts = self._parts()
            if not parts:
                raise SessionError("not found", 404)
            self._authorize(parts=parts)
            if parts == ["tasks"] and method == "GET":
                return {"tasks": sessions.listed()}
            if parts == ["scenes"] and method == "GET":
                return {"scenes": sessions.free_play()}
            if parts == ["sessions"] and method == "POST":
                body = self._body()
                session = sessions.open(task_id=body.get("task_id"), game=body.get("game"),
                                        init_state=body.get("init_state"),
                                        want_state=bool(body.get("state", False)), run=body.get("run"))
                return {**session.describe(), "token": session.token}
            if parts[0] == "sessions" and len(parts) in (2, 3):
                session_id = parts[1]
                if method == "DELETE" and len(parts) == 2:
                    sessions.close(session_id=session_id)
                    return {"closed": session_id}
                session = sessions.get(session_id=session_id)
                tail = parts[2] if len(parts) == 3 else "info"
                with session.lock:
                    if session.closed:
                        raise SessionError("no such session", 404)
                    if tail == "info" and method == "GET":
                        return session.describe()
                    if tail == "observe" and method in ("GET", "POST"):
                        return session.observe()
                    if method != "POST":
                        raise SessionError("not found", 404)
                    if tail == "reset":
                        return session.reset()
                    if tail == "step":
                        action = self._body().get("action")
                        if not isinstance(action, str):
                            raise SessionError("step needs an action string")
                        return session.step(action=action)
                    if tail == "save":
                        return session.save(name=self._body().get("name"))
                    if tail == "checkpoint":
                        label = self._body().get("label")
                        if not isinstance(label, str) or not label:
                            raise SessionError("checkpoint needs a label")
                        return session.mark(label=label)
                    if tail == "load":
                        handle = self._body().get("handle")
                        if not isinstance(handle, str):
                            raise SessionError("load needs a handle")
                        return session.load(handle=handle)
            raise SessionError("not found", 404)

        def _dispatch(self, method: str) -> None:
            try:
                self._send(self._route(method))
            except SessionError as exc:
                self._send({"error": str(exc)}, exc.status)
            except Exception:
                # Exception text can carry host paths, so the client gets only an id to quote.
                incident = uuid.uuid4().hex[:12]
                log_warn(f"internal error {incident} on {method} {self.path[:200]}:\n{traceback.format_exc()}",
                         parameters=parameters)
                self._send({"error": "internal error", "incident": incident}, 500)

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def do_DELETE(self):
            self._dispatch("DELETE")

    return Handler


def _terminate(signum, frame):
    raise KeyboardInterrupt


def serve(*, sessions: Sessions, host: str, port: Optional[int], token: str, parameters: dict,
          unix_socket: Optional[str] = None) -> None:
    handler = _handler(sessions=sessions, token=token, parameters=parameters)
    if unix_socket:
        httpd = unix_listener(unix_socket, handler)
        where = f"unix:{unix_socket}"
    else:
        httpd = ThreadingHTTPServer((host, port), handler)
        httpd.daemon_threads = True
        where = f"{host}:{port}"
    log_info(f"env server ({sessions.policy.kind}) on {where}", parameters=parameters)
    signal.signal(signal.SIGTERM, _terminate)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    stop = threading.Event()
    try:
        while not stop.wait(REAP_EVERY_SECONDS):
            sessions.reap()
    except KeyboardInterrupt:
        log_info("shutting down", parameters=parameters)
    finally:
        httpd.shutdown()
        httpd.server_close()
        sessions.close_all()
        if unix_socket and os.path.exists(unix_socket):
            os.unlink(unix_socket)
