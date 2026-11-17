import http.client
import json
import os
import socket
from socketserver import ThreadingMixIn, UnixStreamServer
from typing import Optional


class UnixHTTPServer(ThreadingMixIn, UnixStreamServer):
    daemon_threads = True

    def get_request(self):
        # BaseHTTPRequestHandler expects a (host, port) peer address; a Unix peer has none.
        request, _ = super().get_request()
        return request, ("unix", 0)


def unix_listener(path: str, handler) -> UnixHTTPServer:
    if os.path.exists(path):
        os.unlink(path)
    # Bound under a restrictive umask so the socket is never connectable by another user.
    previous = os.umask(0o177)
    try:
        return UnixHTTPServer(path, handler)
    finally:
        os.umask(previous)


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


def unix_request(*, socket_path: str, method: str, path: str, token: str, body: Optional[dict] = None,
                 timeout: float = 300.0) -> tuple:
    connection = _UnixHTTPConnection(socket_path, timeout)
    try:
        connection.request(method, path, body=None if body is None else json.dumps(body).encode(),
                           headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read() or b"{}")
    finally:
        connection.close()
