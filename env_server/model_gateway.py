"""
The sandbox's model access: an OpenAI-compatible endpoint on a Unix socket that forwards to OpenRouter.

The sandbox runs in an empty network namespace. Its only ways out are two bind-mounted Unix sockets: the env server
and this gateway. prime-agent is pointed here by a custom provider in ~/.prime/agent/models.json, through
containers/netbridge.py, and so never holds the API key -- the key is read from the host environment here and
added on the way out. A key that is not in the sandbox cannot be read, printed or sent anywhere by the agent.

What it forwards is narrower than what OpenRouter offers, because what OpenRouter offers includes the web:

- only POST /v1/chat/completions, and only for allowlisted model ids. That excludes `:online` variants and models
  that search natively, and makes the model under test the model the run says it is;
- `plugins` and `web_search_options` are removed from every request -- OpenRouter's web search lives there -- as
  are `models` and `route`, its fallback routing, which would otherwise reach models off the allowlist.

Everything else -- messages, images, tools, streaming -- passes through untouched, and the response is streamed
back as it arrives. Every request is logged with its model, its verdict, anything stripped, the upstream status and
byte counts; never its content. Stripped fields are logged by name because an agent sending them is an agent trying
to reach past the model, which the stress tests look for.

    python -m env_server.model_gateway --socket <path> --allow google/gemini-3.6-flash --log <file>
"""
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
#: Request fields that reach past the model itself.
STRIPPED = ("plugins", "web_search_options", "models", "route")
UPSTREAM_TIMEOUT = 600
MAX_BODY = 64 * 1024 * 1024


def _tls_context() -> ssl.SSLContext:
    """Verified TLS that does not depend on where this interpreter's OpenSSL looks for CAs. The project venv's
    Python finds none on this cluster (CERTIFICATE_VERIFY_FAILED); certifi's bundle is what httpx and openai
    already verify against here."""
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
            length = int(self.headers.get("Content-Length") or 0)
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
                # The detail goes to the host log only: an upstream error can carry host details the agent
                # has no business seeing.
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
