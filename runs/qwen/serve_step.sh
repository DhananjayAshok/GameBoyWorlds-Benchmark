#!/bin/bash
shopt -s expand_aliases
source ~/.bashrc
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
export TMPDIR="${TMPDIR:-/scratch1/$USER/tmp}"
mkdir -p "$TMPDIR"

echo "host=$(hostname -s) job=$SLURM_JOB_ID step=$SLURM_STEP_ID start=$(date)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

MODEL="$1"; shift
PORT="${PORT:-8000}"
bash ~/vllm_scripts/serve_vllm_auto.sh "$MODEL" --port "$PORT" "$@" || { echo "vLLM failed to start: $MODEL"; exit 1; }

PID_FILE=~/vllm_state/vllm_$(hostname -s)_${PORT}.pid
PID=$(cat "$PID_FILE")
trap "bash ~/vllm_scripts/stop_vllm.sh $PORT" EXIT TERM INT
echo "vLLM healthy: $MODEL pid=$PID at $(date)"
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "vLLM process $PID exited at $(date)"
