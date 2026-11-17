#!/bin/bash
# Usage: bash scripts/core/serve_vllm.sh [MODEL] [VLLM_ARGS...]
# Brings up a vLLM server and blocks until it is healthy. Routes through
# ~/vllm_scripts/serve_vllm_auto.sh if present, otherwise runs `vllm serve` from PATH.

if [ -f "$HOME/vllm_scripts/serve_vllm_auto.sh" ]; then
    echo "Found ~/vllm_scripts; routing through serve_vllm_auto.sh"
    exec bash "$HOME/vllm_scripts/serve_vllm_auto.sh" "$@"
fi

echo "~/vllm_scripts not found; launching \`vllm serve\` directly from the current environment"

if ! command -v vllm > /dev/null 2>&1; then
    echo "❌ ERROR: vllm not found on PATH. Activate the environment that has vLLM installed,"
    echo "   or create ~/vllm_scripts/serve_vllm_auto.sh with your environment-specific setup."
    exit 1
fi

PORT=8000
TIMEOUT=1800  # 20+ min: hybrid/Mamba models can take ~16+ min to warm up

ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[i]}" == "--port" ]]; then
        PORT="${ARGS[i+1]}"
    elif [[ "${ARGS[i]}" =~ --port=(.*) ]]; then
        PORT="${BASH_REMATCH[1]}"
    fi
done

HAS_TP=0
for arg in "$@"; do
    case "$arg" in
        -tp|--tensor-parallel-size|-tp=*|--tensor-parallel-size=*) HAS_TP=1 ;;
    esac
done

if [ "$HAS_TP" -eq 0 ]; then
    NUM_GPUS=""
    if [ -n "$CUDA_VISIBLE_DEVICES" ] && [ "$CUDA_VISIBLE_DEVICES" != "-1" ]; then
        NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
    elif command -v nvidia-smi > /dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -c .)
    fi

    if [ -n "$NUM_GPUS" ] && [ "$NUM_GPUS" -gt 0 ] 2>/dev/null; then
        ARGS+=(-tp "$NUM_GPUS")
        echo "No -tp given; defaulting to visible GPU count: $NUM_GPUS"
    else
        echo "No -tp given and no GPUs detected; leaving vLLM's default (-tp 1)"
    fi
fi

STATE_DIR="${VLLM_STATE_DIR:-$HOME/vllm_state}"
mkdir -p "$STATE_DIR"
HOST=$(hostname -s)
LOG_FILE="$STATE_DIR/vllm_${HOST}_${PORT}.log"
PID_FILE="$STATE_DIR/vllm_${HOST}_${PORT}.pid"

echo "Launching vLLM on port $PORT (host $HOST)..."
echo "  log: $LOG_FILE"
echo "  pid: $PID_FILE"

# Own session/process group, so the whole tree (including the workers holding VRAM) can be killed together.
setsid nohup vllm serve "${ARGS[@]}" > "$LOG_FILE" 2>&1 &
VLLM_PID=$!

END_TIME=$((SECONDS + TIMEOUT))
while [ $SECONDS -lt $END_TIME ]; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "❌ ERROR: vLLM process crashed. Check $LOG_FILE"
        exit 1
    fi

    if curl -s -f http://localhost:${PORT}/health > /dev/null; then
        echo "✅ SUCCESS: vLLM is healthy on port $PORT (host $HOST, pid $VLLM_PID)!"
        echo $VLLM_PID > "$PID_FILE"
        echo "   Stop it with: bash scripts/core/stop_vllm.sh $PORT"
        exit 0
    fi

    sleep 2
done

echo "❌ ERROR: Timed out waiting for vLLM to start on port $PORT. Check $LOG_FILE"
# Never group-kill our own group (would take down an interactive shell).
PGID=$(ps -o pgid= -p "$VLLM_PID" | tr -d ' ')
MYPGID=$(ps -o pgid= -p $$ | tr -d ' ')
if [ -n "$PGID" ] && [ "$PGID" != "$MYPGID" ]; then
    kill -TERM -"$PGID" 2>/dev/null
    sleep 5
    kill -KILL -"$PGID" 2>/dev/null
else
    kill -TERM "$VLLM_PID" 2>/dev/null
    sleep 5
    kill -KILL "$VLLM_PID" 2>/dev/null
fi
exit 1
