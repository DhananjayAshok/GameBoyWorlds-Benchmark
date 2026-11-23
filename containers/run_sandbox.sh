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
#   playthrough  one whole game. --run_dir keeps the emulator checkpoint and the agent's files, so
#                running the same line again continues the same session:
#     bash containers/run_sandbox.sh --kind playthrough --task_id gb.pokemon_brown.playthrough \
#         --run_dir <dir> --model <model> -- <command>
#
#   --harness copies this repo's strategist harness to /work/harness (its state goes to /work/storage):
#     bash containers/run_sandbox.sh --kind playthrough --task_id gb.pokemon_brown.playthrough \
#         --run_dir <dir> --model <model> --harness -- /opt/harness/bin/python /work/harness/run_playthrough.py \
#         --remote --name <name> --model <model> --vlm_kind gateway
#
#   train  harness development: the agent opens the train_sandbox_games playthroughs itself, with named
#          saves. /work/harness is a clone of --harness_repo (harnesses/strategist by default; see
#          containers/harness_repo.sh) on its own branch; each launch merges the repo's main into it.
#          The server logs each session's progress from game memory to /work/obs/progress. On the host,
#          containers/train_watch.sh mirrors the agent's commits (and pushes them with --push_url) and
#          archives its artifacts. containers/run_train_agent.sh runs prime-agent in it:
#     bash containers/run_sandbox.sh --kind train --run_dir <dir> --model <model> --max_cost 1000 -- <command>
#
# Every kind writes the sessions' videos to /work/obs, read-only. --max_cost caps the model spend in dollars.
# Needs OPENROUTER_API_KEY in the host environment; it is never passed into the sandbox.
# Exits with the command's status. Logs, and result.json for a test or playthrough run, go to --log_dir.

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
MAX_STEPS=""
STATE_DIR=""
CHECKPOINT_EVERY=500
HARNESS=false
MAX_COST=""
HARNESS_REPO=harnesses/strategist
PUSH_URL=""
PUSH_PREFIX=""
STOP_AFTER_NO_PROGRESS=""

usage() {
    sed -n '2,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    echo
    echo "Options: --kind dev|test|playthrough|train  --model <openrouter id>  --task_id <id>  --sif <image>"
    echo "         --workspace <dir>  --log_dir <dir>  --record_dir <dir>  --time_limit <seconds>"
    echo "         --max_steps <n>  --run_dir <dir>  --checkpoint_every <steps>  --harness  --max_cost <dollars>"
    echo "         --harness_repo <dir>  --push_url <git url>  --push_prefix <branch prefix>   (train)"
    echo "         --stop_after_no_progress <steps>   (test, playthrough: end the episode when game memory shows"
    echo "                                             no new map, event or flag for this many steps)"
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
        --run_dir) STATE_DIR="$2"; shift 2 ;;
        --checkpoint_every) CHECKPOINT_EVERY="$2"; shift 2 ;;
        --harness) HARNESS=true; shift ;;
        --max_cost) MAX_COST="$2"; shift 2 ;;
        --harness_repo) HARNESS_REPO="$2"; shift 2 ;;
        --push_url) PUSH_URL="$2"; shift 2 ;;
        --push_prefix) PUSH_PREFIX="$2"; shift 2 ;;
        --stop_after_no_progress) STOP_AFTER_NO_PROGRESS="$2"; shift 2 ;;
        --) shift; break ;;
        *) usage ;;
    esac
done

[[ $# -gt 0 ]] || usage
[[ "$KIND" == dev || "$KIND" == test || "$KIND" == playthrough || "$KIND" == train ]] \
    || die "--kind must be dev, test, playthrough or train"
[[ -n "$MODEL" ]] || die "--model is required"
OPENS_OWN=false
[[ "$KIND" == dev || "$KIND" == train ]] && OPENS_OWN=true
[[ "$OPENS_OWN" == true || -n "$TASK_ID" ]] || die "--kind $KIND needs --task_id"
[[ "$OPENS_OWN" != true || -z "$TASK_ID" ]] || die "--task_id is not for --kind $KIND; the agent opens sessions itself"
[[ "$KIND" != playthrough && "$KIND" != train || -n "$STATE_DIR" ]] \
    || die "--kind $KIND needs --run_dir, where the run is kept"
[[ "$KIND" == playthrough || "$KIND" == train || -z "$STATE_DIR" ]] \
    || die "--run_dir is only for --kind playthrough or train"
[[ "$KIND" != train || "$HARNESS" == false ]] || die "--kind train always copies the harness; drop --harness"
[[ -z "$STOP_AFTER_NO_PROGRESS" || "$KIND" == test || "$KIND" == playthrough ]] \
    || die "--stop_after_no_progress is for evaluation runs (--kind test or playthrough) only"
[[ "$KIND" != train || -d "$HARNESS_REPO/.git" ]] \
    || die "no harness repository at $HARNESS_REPO; make it with containers/harness_repo.sh sync"
[[ -z "$PUSH_URL" || -n "$PUSH_PREFIX" ]] || die "--push_url needs --push_prefix"
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
if [[ -n "$STATE_DIR" ]]; then
    mkdir -p "$STATE_DIR"
    STATE_DIR="$(cd "$STATE_DIR" && pwd)"
    exec 9> "$STATE_DIR/lock"
    flock -n 9 || die "$STATE_DIR is in use by another run"
    AGENT_HOME="$STATE_DIR/home"
    [[ -n "$WORKSPACE" ]] || WORKSPACE="$STATE_DIR/work"
    [[ -n "$LOG_DIR" ]] || LOG_DIR="$STATE_DIR/logs/$(date +%Y%m%d_%H%M%S)"
fi
[[ -n "$WORKSPACE" ]] || WORKSPACE="$RUN_DIR/work"
[[ -n "$LOG_DIR" ]] || LOG_DIR="$results_dir/sandbox/$(date +%Y%m%d_%H%M%S)_$KIND"
OBS_DIR="${STATE_DIR:-$LOG_DIR}/obs"
LEDGER="${STATE_DIR:-$LOG_DIR}/spend.json"
mkdir -p "$NET" "$AGENT_HOME/.prime/agent" "$WORKSPACE/obs" "$LOG_DIR" "$OBS_DIR" "$RUN_DIR/empty"

SERVER_PID=""
GATEWAY_PID=""
WATCH_PID=""
cleanup() {
    [[ -n "$WATCH_PID" ]] && kill -- -"$WATCH_PID" 2>/dev/null
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
    [[ -n "$GATEWAY_PID" ]] && kill "$GATEWAY_PID" 2>/dev/null
    wait 2>/dev/null
    [[ -n "$WATCH_PID" ]] && bash containers/train_watch.sh once "$STATE_DIR" "$PUSH_URL" "$PUSH_PREFIX"
    rm -rf "$RUN_DIR"
}
trap cleanup EXIT

"$PYTHON" -c 'import secrets; print(secrets.token_urlsafe(24))' > "$TOKEN_FILE"

server_args=(serve --kind "$KIND" --unix_socket "$NET/envserver.sock" --token_file "$TOKEN_FILE"
             --obs_dir "$OBS_DIR")
[[ -n "$MAX_STEPS" ]] && server_args+=(--max_steps "$MAX_STEPS")
[[ -n "$RECORD_DIR" ]] && server_args+=(--record_dir "$RECORD_DIR")
[[ "$KIND" == playthrough ]] && server_args+=(--checkpoint_dir "$STATE_DIR/checkpoint" --checkpoint_every "$CHECKPOINT_EVERY")
[[ "$KIND" == train ]] && server_args+=(--saves_dir "$STATE_DIR/saves" --max_sessions 1
                                         --progress_dir "$OBS_DIR/progress")
[[ -n "$STOP_AFTER_NO_PROGRESS" ]] && server_args+=(--progress_dir "$OBS_DIR/progress"
                                                    --stop_after_no_progress "$STOP_AFTER_NO_PROGRESS")
"$PYTHON" run_env_server.py "${server_args[@]}" > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
gateway_args=(--socket "$NET/model.sock" --allow "$MODEL" --log "$LOG_DIR/gateway.jsonl")
[[ -n "$MAX_COST" ]] && gateway_args+=(--max_cost "$MAX_COST" --ledger "$LEDGER")
"$PYTHON" -m env_server.model_gateway "${gateway_args[@]}" > "$LOG_DIR/gateway.log" 2>&1 &
GATEWAY_PID=$!
wait_for_socket "$NET/envserver.sock" "$SERVER_PID" "$LOG_DIR/server.log"
wait_for_socket "$NET/model.sock" "$GATEWAY_PID" "$LOG_DIR/gateway.log"

cp agent_workspace/envclient.py agent_workspace/envctl "$WORKSPACE/"
HARNESS_DIR="$WORKSPACE/harness"
copy_harness() {
    mkdir -p "$HARNESS_DIR/configs" "$HARNESS_DIR/agent_workspace"
    rsync -a --delete --exclude __pycache__ execution utils python_scripts run_playthrough.py "$HARNESS_DIR/"
    cp agent_workspace/envclient.py "$HARNESS_DIR/agent_workspace/"
    "$PYTHON" - "$HARNESS_DIR/configs" <<'PY'
import sys, yaml
out = sys.argv[1]
project = yaml.safe_load(open("configs/project_vars.yaml"))
project["results_dir"] = "/work/results"
yaml.safe_dump(project, open(f"{out}/project_vars.yaml", "w"))
yaml.safe_dump({"storage_dir": "/work/storage"}, open(f"{out}/private_vars.yaml", "w"))
PY
}
if [[ "$HARNESS" == true ]]; then
    copy_harness
fi
if [[ "$KIND" == train ]]; then
    touch "$WORKSPACE/AGENTS.md"
    mkdir -p "$WORKSPACE/agent_logs" "$OBS_DIR/progress"
    bash containers/harness_repo.sh prepare "$HARNESS_REPO" "$HARNESS_DIR" "agent/$(basename "$STATE_DIR")" \
        "$WORKSPACE/agent_logs/next_prompt_note.md"
    export APPTAINERENV_TRAIN_GAMES="$("$PYTHON" -c 'from env_server import splits; from utils import load_parameters; print(" ".join(splits.train_sandbox_games(parameters=load_parameters())))')"
    setsid bash containers/train_watch.sh loop "$STATE_DIR" "$PUSH_URL" "$PUSH_PREFIX" 9>&- &
    WATCH_PID=$!
fi
cat > "$AGENT_HOME/.prime/agent/models.json" <<JSON
{ "providers": { "gateway": {
    "baseUrl": "http://127.0.0.1:$MODEL_GATEWAY_PORT/v1", "apiKey": "none", "api": "openai-completions",
    "models": [ { "id": "$MODEL", "input": ["text", "image"], "reasoning": false,
                  "contextWindow": 1000000, "maxTokens": 65536 } ] } } }
JSON

json_field() { "$PYTHON" -c 'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"; }

SESSION=""
if [[ "$OPENS_OWN" != true ]]; then
    opened="$("$PYTHON" run_env_server.py open_session --unix_socket "$NET/envserver.sock" \
        --token_file "$TOKEN_FILE" --task_id "$TASK_ID")"
    SESSION="$(json_field session <<< "$opened")"
    export APPTAINERENV_ENV_SERVER_TOKEN="$(json_field token <<< "$opened")"
    export APPTAINERENV_ENVCTL_SESSION="$SESSION"
else
    export APPTAINERENV_ENV_SERVER_TOKEN="$(cat "$TOKEN_FILE")"
fi
export APPTAINERENV_ENV_SERVER_URL="http://127.0.0.1:$ENV_SERVER_PORT"
export APPTAINERENV_MODEL_GATEWAY_URL="http://127.0.0.1:$MODEL_GATEWAY_PORT/v1"
export APPTAINERENV_SANDBOX_BRIDGE="$ENV_SERVER_PORT=/sandboxnet/envserver.sock $MODEL_GATEWAY_PORT=/sandboxnet/model.sock"
export APPTAINERENV_SANDBOX_MODEL="$MODEL"

binds=(--bind "$WORKSPACE":/work --bind "$NET":/sandboxnet --bind "$OBS_DIR":/work/obs:ro)
if [[ "$KIND" == train ]]; then
    binds+=(--bind "$PWD/containers/train_agent":/opt/train:ro
            --bind "$PWD/containers/train_agent/AGENTS.md":/work/AGENTS.md:ro)
fi

limit=()
if [[ -n "$TIME_LIMIT" ]]; then
    export APPTAINERENV_ENVCTL_DEADLINE="$(( $(date +%s) + TIME_LIMIT ))"
    limit=(timeout --signal=TERM --kill-after=30 "$TIME_LIMIT")
fi

status=0
"${limit[@]}" apptainer exec "${SANDBOX_FLAGS[@]}" \
    --home "$AGENT_HOME":/home/agent "${binds[@]}" --pwd /work \
    "$SIF" sandbox-run "$@" || status=$?

if [[ -n "$SESSION" ]]; then
    "$PYTHON" run_env_server.py session_info --unix_socket "$NET/envserver.sock" \
        --token_file "$TOKEN_FILE" --session "$SESSION" > "$LOG_DIR/result.json"
    echo "result: $(cat "$LOG_DIR/result.json")"
fi
echo "logs: $LOG_DIR"
exit "$status"
