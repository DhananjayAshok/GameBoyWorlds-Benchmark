#!/usr/bin/env bash
# Run a command inside the isolated sandbox, with an env server and a model gateway started for it.
#
#   dev   the agent may list and open train/val tasks itself:
#     bash containers/run_sandbox.sh --kind dev --model google/gemini-3.6-flash -- \
#         prime-agent --provider gateway --model google/gemini-3.6-flash --cwd /work -p "<prompt>"
#
#   test  one held-out task, opened for the agent; it gets a token for that session only:
#     bash containers/run_sandbox.sh --kind test --task_id gb.pokemon_red.001 --model <model> -- <command>
#
# Needs OPENROUTER_API_KEY in the host environment; it is never passed into the sandbox.
# Exits with the command's status. Logs, and result.json for a test run, go to --log_dir.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source configs/config.env
source containers/common.sh

KIND=dev
MODEL=""
TASK_ID=""
SIF="$(default_sif)"
WORKSPACE=""
LOG_DIR=""
RECORD_DIR=""
TIME_LIMIT=""
MAX_STEPS=200

usage() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    echo
    echo "Options: --kind dev|test  --model <openrouter id>  --task_id <id>  --sif <image>"
    echo "         --workspace <dir>  --log_dir <dir>  --record_dir <dir>  --time_limit <seconds>"
    echo "         --max_steps <n>"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kind) KIND="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --task_id) TASK_ID="$2"; shift 2 ;;
        --sif) SIF="$2"; shift 2 ;;
        --workspace) WORKSPACE="$2"; shift 2 ;;
        --log_dir) LOG_DIR="$2"; shift 2 ;;
        --record_dir) RECORD_DIR="$2"; shift 2 ;;
        --time_limit) TIME_LIMIT="$2"; shift 2 ;;
        --max_steps) MAX_STEPS="$2"; shift 2 ;;
        --) shift; break ;;
        *) usage ;;
    esac
done

[[ $# -gt 0 ]] || usage
[[ "$KIND" == dev || "$KIND" == test ]] || die "--kind must be dev or test"
[[ -n "$MODEL" ]] || die "--model is required"
[[ "$KIND" == dev || -n "$TASK_ID" ]] || die "--kind test needs --task_id"
[[ "$KIND" == test || -z "$TASK_ID" ]] || die "--task_id is only for --kind test; a dev agent opens tasks itself"
[[ -f "$SIF" ]] || die "no image at $SIF; build it first (see containers/sandbox.def)"
[[ -n "${OPENROUTER_API_KEY:-}" ]] || die "OPENROUTER_API_KEY is not set; the model gateway needs it"
require_apptainer

PYTHON=setup/.venv/bin/python
[[ -x "$PYTHON" ]] || die "no virtual environment at setup/.venv; run setup/create_env.sh"

umask 077
RUN_DIR="$(mktemp -d "$(sandbox_runtime_dir)/gbsandbox.XXXXXX")"
NET="$RUN_DIR/net"
AGENT_HOME="$RUN_DIR/home"
TOKEN_FILE="$RUN_DIR/server.token"
[[ -n "$WORKSPACE" ]] || WORKSPACE="$RUN_DIR/work"
[[ -n "$LOG_DIR" ]] || LOG_DIR="$results_dir/sandbox/$(date +%Y%m%d_%H%M%S)_$KIND"
mkdir -p "$NET" "$AGENT_HOME/.prime/agent" "$WORKSPACE" "$LOG_DIR"

SERVER_PID=""
GATEWAY_PID=""
cleanup() {
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
    [[ -n "$GATEWAY_PID" ]] && kill "$GATEWAY_PID" 2>/dev/null
    wait 2>/dev/null
    rm -rf "$RUN_DIR"
}
trap cleanup EXIT

"$PYTHON" -c 'import secrets; print(secrets.token_urlsafe(24))' > "$TOKEN_FILE"

server_args=(serve --kind "$KIND" --unix_socket "$NET/envserver.sock" --token_file "$TOKEN_FILE"
             --max_steps "$MAX_STEPS")
[[ -n "$RECORD_DIR" ]] && server_args+=(--record_dir "$RECORD_DIR")
"$PYTHON" run_env_server.py "${server_args[@]}" > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
"$PYTHON" -m env_server.model_gateway --socket "$NET/model.sock" --allow "$MODEL" \
    --log "$LOG_DIR/gateway.jsonl" > "$LOG_DIR/gateway.log" 2>&1 &
GATEWAY_PID=$!
wait_for_socket "$NET/envserver.sock" "$SERVER_PID" "$LOG_DIR/server.log"
wait_for_socket "$NET/model.sock" "$GATEWAY_PID" "$LOG_DIR/gateway.log"

cp agent_workspace/envclient.py agent_workspace/envctl "$WORKSPACE/"
cat > "$AGENT_HOME/.prime/agent/models.json" <<JSON
{ "providers": { "gateway": {
    "baseUrl": "http://127.0.0.1:$MODEL_GATEWAY_PORT/v1", "apiKey": "none", "api": "openai-completions",
    "models": [ { "id": "$MODEL", "input": ["text", "image"], "reasoning": false,
                  "contextWindow": 1000000, "maxTokens": 65536 } ] } } }
JSON

json_field() { "$PYTHON" -c 'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"; }

SESSION=""
if [[ "$KIND" == test ]]; then
    opened="$("$PYTHON" run_env_server.py open_session --unix_socket "$NET/envserver.sock" \
        --token_file "$TOKEN_FILE" --task_id "$TASK_ID")"
    SESSION="$(json_field session <<< "$opened")"
    export APPTAINERENV_ENV_SERVER_TOKEN="$(json_field token <<< "$opened")"
    export APPTAINERENV_ENVCTL_SESSION="$SESSION"
else
    export APPTAINERENV_ENV_SERVER_TOKEN="$(cat "$TOKEN_FILE")"
fi
export APPTAINERENV_ENV_SERVER_URL="http://127.0.0.1:$ENV_SERVER_PORT"
export APPTAINERENV_SANDBOX_BRIDGE="$ENV_SERVER_PORT=/sandboxnet/envserver.sock $MODEL_GATEWAY_PORT=/sandboxnet/model.sock"

limit=()
if [[ -n "$TIME_LIMIT" ]]; then
    export APPTAINERENV_ENVCTL_DEADLINE="$(( $(date +%s) + TIME_LIMIT ))"
    limit=(timeout --signal=TERM --kill-after=30 "$TIME_LIMIT")
fi

status=0
"${limit[@]}" apptainer exec "${SANDBOX_FLAGS[@]}" \
    --home "$AGENT_HOME":/home/agent --bind "$WORKSPACE":/work --bind "$NET":/sandboxnet --pwd /work \
    "$SIF" sandbox-run "$@" || status=$?

if [[ -n "$SESSION" ]]; then
    "$PYTHON" run_env_server.py session_info --unix_socket "$NET/envserver.sock" \
        --token_file "$TOKEN_FILE" --session "$SESSION" > "$LOG_DIR/result.json"
    echo "result: $(cat "$LOG_DIR/result.json")"
fi
echo "logs: $LOG_DIR"
exit "$status"
