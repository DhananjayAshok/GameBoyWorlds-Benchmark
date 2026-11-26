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

PATH = "/v1/chat/completions"
# Request fields that reach past the allowlisted model: web search, and fallback routing to other models.
STRIPPED = ("plugins", "web_search_options", "models", "route")
UPSTREAM_TIMEOUT = 600
MAX_BODY = 64 * 1024 * 1024
STREAM_TAIL = 1024 * 1024
STREAM_RETRIES = 3
RETRY_DELAY = 5
RETRY_STATUS = (429, 500, 502, 503, 504)

GOOGLE_FIELDS = {"model", "messages", "stream", "stream_options", "tools", "tool_choice", "temperature", "top_p",
                 "max_tokens", "max_completion_tokens", "stop", "n", "response_format", "seed", "reasoning_effort",
                 "frequency_penalty", "presence_penalty"}
GOOGLE_MESSAGE_FIELDS = {"role", "content", "name", "tool_calls", "tool_call_id", "extra_content"}
# Google's documented placeholder for a function call whose thought signature was not kept by the client.
GOOGLE_DUMMY_SIGNATURE = "skip_thought_signature_validator"
GOOGLE_SIGNATURES_KEPT = 50000
# Dollars per million tokens: (in effect from, input, output including thinking, cached input), newest first.
GOOGLE_PRICES = {
    "gemini-4-argon": [("2026-12-01", 4.00, 20.00, 0.20), ("", 2.00, 10.00, 0.10)],
}


class OpenRouter:
    url = "https://openrouter.ai/api/v1/chat/completions"
    key_env = "OPENROUTER_API_KEY"

    def prepare(self, request: dict) -> list:
        return []

    def learn(self, body: bytes, *, stream: bool) -> None:
        pass

    def cost(self, usage, model: str = None) -> float:
        return call_cost(usage)


class Google:
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    key_env = "GEMINI_API_KEY"

    def __init__(self, allow) -> None:
        self.lock = threading.Lock()
        self.signatures = {}
        missing = [m for m in allow if self.upstream_model(m) not in GOOGLE_PRICES]
        if missing:
            raise ValueError(f"no Google price for {missing}; add it to GOOGLE_PRICES so the spending cap counts it")

    @staticmethod
    def upstream_model(model: str) -> str:
        return model.split("/", 1)[1] if model.startswith("google/") else model

    def prepare(self, request: dict) -> list:
        stripped = []
        request["model"] = self.upstream_model(request["model"])
        reasoning = request.pop("reasoning", None)
        if isinstance(reasoning, dict) and reasoning.get("effort") and "reasoning_effort" not in request:
            request["reasoning_effort"] = reasoning["effort"]
        for name in [name for name in request if name not in GOOGLE_FIELDS]:
            request.pop(name)
            stripped.append(name)
        if request.get("stream"):
            request["stream_options"] = {**(request.get("stream_options") or {}), "include_usage": True}
        messages = request.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict):
                    self.prepare_message(message, stripped)
        return stripped

    def prepare_message(self, message: dict, stripped: list) -> None:
        for name in [name for name in message if name not in GOOGLE_MESSAGE_FIELDS]:
            message.pop(name)
            if f"messages[].{name}" not in stripped:
                stripped.append(f"messages[].{name}")
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if isinstance(part, dict) and part.pop("cache_control", None) is not None:
                    if "cache_control" not in stripped:
                        stripped.append("cache_control")
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            google = (call.get("extra_content") or {}).get("google") or {}
            if google.get("thought_signature"):
                continue
            with self.lock:
                signature = self.signatures.get(call.get("id")) or GOOGLE_DUMMY_SIGNATURE
            call["extra_content"] = {"google": {**google, "thought_signature": signature}}

    def learn(self, body: bytes, *, stream: bool) -> None:
        text = body.decode("utf-8", "replace")
        events = []
        if stream:
            for line in text.splitlines():
                if line.startswith("data:"):
                    try:
                        events.append(json.loads(line[len("data:"):].strip()))
                    except json.JSONDecodeError:
                        pass
        else:
            try:
                events.append(json.loads(text))
            except json.JSONDecodeError:
                return
        ids, found = {}, {}
        for event in events:
            if not isinstance(event, dict):
                continue
            for choice in event.get("choices") or []:
                part = choice.get("delta") or choice.get("message") or {}
                for position, call in enumerate(part.get("tool_calls") or []):
                    key = (choice.get("index", 0), call.get("index", position))
                    if call.get("id"):
                        ids[key] = call["id"]
                    signature = ((call.get("extra_content") or {}).get("google") or {}).get("thought_signature")
                    if signature and ids.get(key):
                        found[ids[key]] = signature
        if found:
            with self.lock:
                self.signatures.update(found)
                while len(self.signatures) > GOOGLE_SIGNATURES_KEPT:
                    self.signatures.pop(next(iter(self.signatures)))

    def cost(self, usage, model: str = None) -> float:
        if not isinstance(usage, dict):
            return 0.0
        prompt = int(usage.get("prompt_tokens") or 0)
        total = usage.get("total_tokens")
        output = int(total) - prompt if total is not None else int(usage.get("completion_tokens") or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        today = time.strftime("%Y-%m-%d", time.gmtime())
        _, price_in, price_out, price_cached = next(
            row for row in GOOGLE_PRICES[self.upstream_model(model)] if row[0] <= today)
        return ((prompt - cached) * price_in + cached * price_cached + output * price_out) / 1e6


def stream_error(line: bytes):
    try:
        event = json.loads(line[len(b"data:"):].strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict) or "error" not in event:
        return None
    error = event["error"]
    return str(error.get("message") if isinstance(error, dict) else error)[:200]


def call_cost(usage) -> float:
    if not isinstance(usage, dict):
        return 0.0
    details = usage.get("cost_details") or {}
    return float(usage.get("cost") or 0.0) + float(details.get("upstream_inference_cost") or 0.0)


def usage_of(body: bytes, *, stream: bool):
    text = body.decode("utf-8", "replace")
    if not stream:
        try:
            return json.loads(text).get("usage")
        except (json.JSONDecodeError, AttributeError):
            return None
    for line in reversed(text.splitlines()):
        if line.startswith("data:") and '"usage"' in line:
            try:
                return json.loads(line[len("data:"):].strip()).get("usage")
            except (json.JSONDecodeError, AttributeError):
                return None
    return None


class Ledger:
    def __init__(self, *, path: str, max_cost: float) -> None:
        self.path = path
        self.max_cost = max_cost
        self.lock = threading.Lock()
        self.totals = {"spent": 0.0, "calls": 0, "calls_without_usage": 0}
        if os.path.exists(path):
            with open(path) as f:
                self.totals.update(json.load(f))

    def exhausted(self) -> bool:
        with self.lock:
            return self.totals["spent"] >= self.max_cost

    def add(self, *, usage, cost: float) -> float:
        with self.lock:
            self.totals["spent"] += cost
            self.totals["calls"] += 1
            self.totals["calls_without_usage"] += usage is None
            with open(self.path + ".tmp", "w") as f:
                json.dump({**self.totals, "max_cost": self.max_cost}, f)
            os.replace(self.path + ".tmp", self.path)
            return self.totals["spent"]


def _tls_context() -> ssl.SSLContext:
    # certifi's bundle, because a venv Python does not always find the system CA store.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


TLS = _tls_context()


def _handler(*, allow: set, key: str, log, lock: threading.Lock, ledger=None, provider=None):
    provider = provider or OpenRouter()

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
            if ledger is not None and ledger.exhausted():
                return self._refuse(402, f"the spending cap of ${ledger.max_cost:g} has been reached", model=model)
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
            stripped += provider.prepare(request)

            body = json.dumps(request).encode()
            accept = self.headers.get("Accept", "application/json")

            def open_upstream():
                upstream = urllib.request.Request(
                    provider.url, data=body, method="POST",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                             "Accept": accept})
                try:
                    return urllib.request.urlopen(upstream, timeout=UPSTREAM_TIMEOUT, context=TLS)
                except urllib.error.HTTPError as error:
                    return error

            started, sent, retries = time.time(), 0, 0
            retried = []
            try:
                response = open_upstream()
                while response.status in RETRY_STATUS and retries < STREAM_RETRIES:
                    retried.append(f"HTTP {response.status}")
                    response.close()
                    time.sleep(RETRY_DELAY * 2 ** retries)
                    retries += 1
                    response = open_upstream()
            except Exception as error:
                # The detail goes to the host log only.
                return self._refuse(502, f"upstream unreachable: {type(error).__name__}", model=model,
                                    stripped=stripped, detail=str(getattr(error, "reason", error))[:300])
            # HTTP/1.0 semantics (the handler's default): the body runs to the close, so a stream needs no framing.
            self.send_response(response.status)
            self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
            self.end_headers()
            stream = bool(request.get("stream"))
            kept = bytearray()
            client_gone = False

            def forward(data: bytes) -> None:
                nonlocal sent, client_gone
                if not data or client_gone:
                    return
                try:
                    self.wfile.write(data)
                    self.wfile.flush()
                    sent += len(data)
                except (BrokenPipeError, ConnectionResetError):
                    client_gone = True

            while True:
                content, pending, withheld = not stream, bytearray(), None
                try:
                    while True:
                        chunk = response.read1(65536) if hasattr(response, "read1") else response.read(65536)
                        if not chunk:
                            break
                        kept += chunk
                        if stream and len(kept) > STREAM_TAIL:
                            del kept[:len(kept) - STREAM_TAIL]
                        if content:
                            forward(chunk)
                            continue
                        pending += chunk
                        out = bytearray()
                        while b"\n" in pending:
                            end = pending.index(b"\n") + 1
                            line = bytes(pending[:end])
                            del pending[:end]
                            if line.startswith(b"data:"):
                                if stream_error(line) is not None:
                                    withheld = line
                                    break
                                content = True
                                out += line + pending
                                pending.clear()
                                break
                            out += line
                        forward(bytes(out))
                        if withheld is not None:
                            break
                finally:
                    response.close()
                if withheld is None and pending:
                    if pending.startswith(b"data:") and stream_error(bytes(pending)) is not None:
                        withheld = bytes(pending)
                    else:
                        content = content or pending.startswith(b"data:")
                        forward(bytes(pending))
                if content or client_gone or response.status != 200 or retries >= STREAM_RETRIES:
                    forward(withheld or b"")
                    break
                retried.append(stream_error(withheld) if withheld is not None else "no data")
                retries += 1
                time.sleep(RETRY_DELAY)
                try:
                    response = open_upstream()
                except Exception:
                    forward(withheld or b"")
                    break
            usage = usage_of(bytes(kept), stream=stream) if response.status == 200 else None
            if response.status == 200:
                provider.learn(bytes(kept), stream=stream)
            cost = provider.cost(usage, model)
            spent = ledger.add(usage=usage, cost=cost) if ledger is not None and response.status == 200 else None
            record(verdict="forwarded", model=model, stripped=stripped, status=response.status,
                   stream=stream, bytes_in=length, bytes_out=sent, seconds=round(time.time() - started, 2),
                   cost=cost, spent=spent, retries=retries, retried=retried,
                   **({"tail": bytes(kept[-800:]).decode("utf-8", "replace")}
                      if (stream and usage is None) or response.status != 200 else {}))

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible endpoint on a Unix socket that forwards chat completions for "
                    "allowlisted models to OpenRouter or Google, adding the API key on the host.")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--allow", action="append", required=True, help="an OpenRouter model id, repeatable")
    parser.add_argument("--log", required=True)
    parser.add_argument("--provider", choices=("openrouter", "google"), default="openrouter")
    parser.add_argument("--key_env", default=None,
                        help="the host environment variable holding the key (default: the provider's); "
                             "it is never written anywhere")
    parser.add_argument("--max_cost", type=float, default=None,
                        help="refuse requests once this many dollars have been spent, counted in --ledger")
    parser.add_argument("--ledger", default=None, help="a JSON file that keeps the running spend across restarts")
    args = parser.parse_args()
    allow = set(args.allow)
    if any(m.endswith(":online") for m in allow):
        parser.error("an :online model searches the web; it cannot be allowlisted")
    try:
        provider = Google(allow) if args.provider == "google" else OpenRouter()
    except ValueError as error:
        parser.error(str(error))
    key_env = args.key_env or provider.key_env
    key = os.environ.get(key_env)
    if not key:
        parser.error(f"${key_env} is not set")
    if (args.max_cost is None) != (args.ledger is None):
        parser.error("--max_cost and --ledger go together")
    ledger = Ledger(path=args.ledger, max_cost=args.max_cost) if args.ledger else None
    with open(args.log, "a") as log:
        lock = threading.Lock()
        httpd = unix_listener(args.socket, _handler(allow=allow, key=key, log=log, lock=lock, ledger=ledger,
                                                    provider=provider))
        log.write(json.dumps({"t": round(time.time(), 3), "verdict": "listening", "socket": args.socket,
                              "provider": args.provider, "allow": sorted(allow), "max_cost": args.max_cost,
                              "spent": ledger.totals["spent"] if ledger else None}) + "\n")
        log.flush()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
