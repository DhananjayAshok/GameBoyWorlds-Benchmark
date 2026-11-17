"""
A threading HTTP server on a Unix socket, shared by the env server and the model gateway.

Kept apart from server.py so the gateway does not import the emulator stack to open a socket.
"""
import os
from socketserver import ThreadingMixIn, UnixStreamServer


class UnixHTTPServer(ThreadingMixIn, UnixStreamServer):
    daemon_threads = True

    def get_request(self):
        # A Unix peer has no address; BaseHTTPRequestHandler expects a (host, port) pair.
        request, _ = super().get_request()
        return request, ("unix", 0)


def unix_listener(path: str, handler) -> UnixHTTPServer:
    """A threading HTTP server on a Unix socket only this user can connect to."""
    if os.path.exists(path):
        os.unlink(path)
    httpd = UnixHTTPServer(path, handler)
    os.chmod(path, 0o600)
    return httpd
