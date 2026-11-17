#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
JOB="$1"; CANCEL="$2"; SERVE_LOG="$3"; WAIT_STEP="$4"; shift 4
if [[ -n "$WAIT_STEP" ]]; then
    stamp "waiting for $JOB.$WAIT_STEP to finish"
    while squeue -h -s -j "$JOB" -o "%i" | grep -qx "$JOB.$WAIT_STEP"; do sleep 120; done
fi
for s in ${CANCEL//,/ }; do
    stamp "scancel $JOB.$s"
    scancel "$JOB.$s"
done
for s in ${CANCEL//,/ }; do
    while squeue -h -s -j "$JOB" -o "%i" | grep -qx "$JOB.$s"; do sleep 10; done
done
sleep 30
stamp "starting server: $*"
CPUS=$(squeue -h -j "$JOB" -o "%C")
bash runs/qwen/launch_step.sh "$JOB" "$CPUS" "$SERVE_LOG" bash runs/qwen/serve_step.sh "$@"
until grep -q "vLLM healthy\|vLLM failed" "$SERVE_LOG" 2>/dev/null; do sleep 30; done
if grep -q "vLLM failed" "$SERVE_LOG"; then
    stamp "SERVER FAILED"
    exit 1
fi
stamp "server healthy"
