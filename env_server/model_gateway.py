import argparse
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

from env_server.unix_http import unix_listener

UPSTREAM = "https://openrouter.ai/api/v1/chat/completions"
PATH = "/v1/chat/completions"
# Request fields that reach past the allowlisted model: web search, and fallback routing to other models.
STRIPPED = ("plugins", "web_search_options", "models", "route")
UPSTREAM_TIMEOUT = 600
MAX_BODY = 64 * 1024 * 1024


def _tls_context() -> ssl.SSLContext:
    # certifi's bundle, because a venv Python does not always find the system CA store.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


TLS = _tls_context()


def _handler(*, allow: set, key: str, log, lock: threading.Lock):
    def record(**event) -> None:
        event["t"] = round(time.time(), 3)
        with lock:
            log.write(json.dumps(event) + "\n")
            log.flush()

    class Handler(BaseHTTPRequestHandler):
        server_version = "sandbox-model-gateway"

        def log_message(self, fmt, *args):
            pass

        def _refuse(self, status: int, message: str, **event) -> None:
            record(verdict="refused", status=status, reason=message, **event)
            body = json.dumps({"error": {"message": message, "code": status}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._refuse(404, "only POST /v1/chat/completions is served", path=self.path[:200])

        def do_POST(self):
            if self.path.split("?")[0] != PATH:
                return self._refuse(404, "only POST /v1/chat/completions is served", path=self.path[:200])
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if not 0 < length <= MAX_BODY:
                return self._refuse(400, "a JSON body is required")
            try:
                request = json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                return self._refuse(400, "the body is not JSON")
            if not isinstance(request, dict):
                return self._refuse(400, "the body is not a JSON object")
            model = request.get("model")
            if model not in allow:
                return self._refuse(403, f"model {model!r} is not allowed", model=str(model)[:200])
            stripped = [name for name in STRIPPED if name in request]
            for name in stripped:
                request.pop(name)
            # Only client-executed function tools pass: a server-side tool runs upstream, with network access.
            tools = request.get("tools")
            if isinstance(tools, list):
                kept = [t for t in tools if isinstance(t, dict) and t.get("type") == "function"]
                if len(kept) != len(tools):
                    request["tools"] = kept
                    stripped.append("tools[server-side]")

            upstream = urllib.request.Request(
                UPSTREAM, data=json.dumps(request).encode(), method="POST",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                         "Accept": self.headers.get("Accept", "application/json")})
            started, sent = time.time(), 0
            try:
                response = urllib.request.urlopen(upstream, timeout=UPSTREAM_TIMEOUT, context=TLS)
            except urllib.error.HTTPError as error:
                response = error
            except Exception as error:
                # The detail goes to the host log only.
                return self._refuse(502, f"upstream unreachable: {type(error).__name__}", model=model,
                                    stripped=stripped, detail=str(getattr(error, "reason", error))[:300])
            # HTTP/1.0 semantics (the handler's default): the body runs to the close, so a stream needs no framing.
            self.send_response(response.status)
            self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
            self.end_headers()
            try:
                while True:
                    chunk = response.read1(65536) if hasattr(response, "read1") else response.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    sent += len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                response.close()
            record(verdict="forwarded", model=model, stripped=stripped, status=response.status,
                   stream=bool(request.get("stream")), bytes_in=length, bytes_out=sent,
                   seconds=round(time.time() - started, 2))

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible endpoint on a Unix socket that forwards chat completions for "
                    "allowlisted models to OpenRouter, adding the API key on the host.")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--allow", action="append", required=True, help="an OpenRouter model id, repeatable")
    parser.add_argument("--log", required=True)
    parser.add_argument("--key_env", default="OPENROUTER_API_KEY",
                        help="the host environment variable holding the key; it is never written anywhere")
    args = parser.parse_args()
    key = os.environ.get(args.key_env)
    if not key:
        parser.error(f"${args.key_env} is not set")
    allow = set(args.allow)
    if any(m.endswith(":online") for m in allow):
        parser.error("an :online model searches the web; it cannot be allowlisted")
    with open(args.log, "a") as log:
        lock = threading.Lock()
        httpd = unix_listener(args.socket, _handler(allow=allow, key=key, log=log, lock=lock))
        log.write(json.dumps({"t": round(time.time(), 3), "verdict": "listening", "socket": args.socket,
                              "allow": sorted(allow)}) + "\n")
        log.flush()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
