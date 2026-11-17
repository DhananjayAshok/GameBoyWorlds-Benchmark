#!/bin/bash
# End to end, isolated: the agent runs in a sandbox with no network, and reaches exactly two things
# through bind-mounted Unix sockets -- the env server, and the model gateway that holds the API key.
#
#     bash containers/smoke_e2e.sh
#
# Needs OPENROUTER_API_KEY in the host environment. It goes to the gateway on the host and never into
# the sandbox. Makes a handful of gemini-3.6-flash calls.
#
# Passing means all of:
#   - escape probes: a host-loopback service, the internet and DNS are all unreachable from inside;
#   - gateway probes: a disallowed model and an :online variant are refused, the web plugin and
#     fallback-model fields are stripped (see the gateway log), GET /v1/models is refused;
#   - the agent opens a real task over the env-server socket, steps it, attaches the PNG frame,
#     reads obs.signals, closes the session, and installs nothing at runtime;
#   - the API key is in neither the sandbox's environment nor any file it wrote.
# Check the SCREEN line against obs/<session>/0001.png yourself: a model will report success it
# did not achieve.
cd "$(dirname "$0")/.." || exit 1
module load apptainer 2>/dev/null
export APPTAINER_CACHEDIR=/scratch1/$USER/apptainer_cache TMPDIR=/scratch1/$USER/tmp
SIF=${SIF:-/project2/jessetho_1732/ashokd/GameBoyRL/containers/sandbox-pa.sif}
MODEL=${MODEL:-google/gemini-3.6-flash}
: "${OPENROUTER_API_KEY:?the gateway needs OPENROUTER_API_KEY on the host}"

# Agent home, workspace and channel sockets in node-local memory.
NET=$(mktemp -d /dev/shm/sandboxnet.XXXXXX); AH=$(mktemp -d /dev/shm/agenthome.XXXXXX); WS=$(mktemp -d /dev/shm/ws.XXXXXX)
LOGS=/scratch1/$USER/tmp/smoke_e2e; rm -rf "$LOGS"; mkdir -p "$LOGS"
TOKENFILE=$(mktemp "$NET/.token.XXXXXX"); chmod 600 "$TOKENFILE"
python3 -c 'import secrets; print(secrets.token_urlsafe(24))' > "$TOKENFILE"

setup/.venv/bin/python run_env_server.py serve --kind dev --unix_socket "$NET/envserver.sock" \
    --token_file "$TOKENFILE" > "$LOGS/server.log" 2>&1 &
SERVER=$!
setup/.venv/bin/python -m env_server.model_gateway --socket "$NET/model.sock" --allow "$MODEL" \
    --log "$LOGS/gateway.log" > "$LOGS/gateway.out" 2>&1 &
GATEWAY=$!
trap 'kill $SERVER $GATEWAY $HOSTSVC 2>/dev/null; wait 2>/dev/null; rm -rf "$NET" "$AH" "$WS"' EXIT
for _ in $(seq 120); do [ -S "$NET/envserver.sock" ] && [ -S "$NET/model.sock" ] && break; sleep 0.5; done
TOKEN=$(cat "$TOKENFILE"); rm "$TOKENFILE"
echo "== host side: $(grep -o 'env server.*' "$LOGS/server.log"); gateway: $([ -S "$NET/model.sock" ] && echo up)"

# A service on the host's loopback, standing in for whatever else runs on the node.
HOSTPORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
python3 -c 'import socket, sys
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", int(sys.argv[1]))); s.listen(8)
while True: s.accept()[0].close()' "$HOSTPORT" &
HOSTSVC=$!

cp user_template/envclient.py user_template/envctl "$WS/"
mkdir -p "$AH/.prime/agent"
cat > "$AH/.prime/agent/models.json" <<JSON
{ "providers": { "gateway": {
    "baseUrl": "http://127.0.0.1:8720/v1", "apiKey": "none", "api": "openai-completions",
    "models": [ { "id": "$MODEL", "input": ["text", "image"], "reasoning": false,
                  "contextWindow": 1000000, "maxTokens": 65536 } ] } } }
JSON

# Deliberately not passed: OPENROUTER_API_KEY.
export APPTAINERENV_SANDBOX_BRIDGE="8710=/sandboxnet/envserver.sock 8720=/sandboxnet/model.sock"
export APPTAINERENV_ENV_SERVER_URL=http://127.0.0.1:8710 APPTAINERENV_ENV_SERVER_TOKEN="$TOKEN"
run() {
    apptainer exec --containall --cleanenv --net --network none \
        --home "$AH":/home/agent --bind "$WS":/work --bind "$NET":/sandboxnet --pwd /work "$SIF" \
        sandbox-run "$@"
}

echo "== escape probes"
run python - "$HOSTPORT" <<'PY'
import socket, sys
def tcp(h, p):
    try: socket.create_connection((h, p), timeout=4).close(); return "REACHABLE"
    except Exception as e: return "unreachable (%s)" % type(e).__name__
print("  a service on the host loopback:", tcp("127.0.0.1", int(sys.argv[1])))
print("  the internet, directly:        ", tcp("1.1.1.1", 443))
try: socket.gethostbyname("openrouter.ai"); print("  DNS:                            RESOLVES")
except Exception as e: print("  DNS:                            fails (%s)" % type(e).__name__)
PY

echo "== gateway probes"
run python - "$MODEL" <<'PY'
import json, sys, urllib.request, urllib.error
def post(path, body, method="POST"):
    req = urllib.request.Request("http://127.0.0.1:8720" + path, method=method,
        data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=120); return r.status
    except urllib.error.HTTPError as e: return e.code
msg = [{"role": "user", "content": "Reply with the single word: pong"}]
print("  allowed model:                ", post("/v1/chat/completions", {"model": sys.argv[1], "messages": msg}))
print("  disallowed model:             ", post("/v1/chat/completions", {"model": "openai/gpt-4o-mini", "messages": msg}))
print("  :online variant:              ", post("/v1/chat/completions", {"model": sys.argv[1] + ":online", "messages": msg}))
print("  web plugin + fallback models: ", post("/v1/chat/completions", {"model": sys.argv[1], "messages": msg,
      "plugins": [{"id": "web"}], "models": ["openai/gpt-4o-mini"], "web_search_options": {}}), "(fields stripped)")
print("  GET /v1/models:               ", post("/v1/models", None, "GET"))
PY

echo "== the task"
export APPTAINERENV_TASK_PROMPT='You are in a workspace with a command-line client called envctl for a Game Boy environment server. Using the shell:
1. python envctl open --task gb.pokemon_red.003
2. python envctl step A
3. Look at the PNG named on the "screen:" line of that last output.
4. python envctl close
Then reply in exactly this format and nothing else:
GOAL: <the goal line>
AGENT_STATE: <agent_state from the signals line>
SCREEN: <one sentence describing the image>'
time run sh -c 'timeout 600 prime-agent --provider gateway --model "$0" --cwd /work -p "$TASK_PROMPT"' "$MODEL" 2>&1 \
    | grep -v "setting up python kernel\|✓ ready" | tee "$LOGS/agent.out" | tail -5
echo "  runtime installs (want 0): $(grep -cE 'Downloading|Installed [0-9]+ packages|Building|Creating virtual environment' "$LOGS/agent.out")"

echo "== the API key"
# By its shape: passing the key, or any part of it, in to search for would put it in the sandbox.
echo "  key-shaped values in the sandbox's environment: $(run sh -c 'env | grep -c "sk-or-"')"
echo "  in files the sandbox wrote:   $(grep -rlF -- "$OPENROUTER_API_KEY" "$AH" "$WS" 2>/dev/null | wc -l)"

echo "== host-side records"
echo "  env server: $(grep -c 'opened session' "$LOGS/server.log") opened, $(grep -c 'Closing environment' "$LOGS/server.log") closed"
python3 - "$LOGS/gateway.log" <<'PY'
import json, sys, collections
c = collections.Counter()
for line in open(sys.argv[1]):
    e = json.loads(line)
    if e["verdict"] == "listening": continue
    c[(e["verdict"], e.get("model") or e.get("path") or "", ",".join(e.get("stripped") or []), e.get("status"))] += 1
for (v, m, s, st), n in sorted(c.items(), key=str):
    print("  gateway %-9s %-32s status=%s stripped=[%s] x%d" % (v, m[:32], st, s, n))
PY
echo "  logs: $LOGS"
