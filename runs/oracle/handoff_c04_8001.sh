#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
JOB=6787924
stamp "waiting for 32B deja_vu lane ($JOB.5) to finish"
while squeue -h -s -j $JOB -o "%i" | grep -qx "$JOB.5"; do
    sleep 120
done
stamp "lane step gone; last lane line: $(tail -1 logs/qwen/cur_32b_deja_vu.out)"
stamp "cancelling 32B server $JOB.2"
scancel $JOB.2
while squeue -h -s -j $JOB -o "%i" | grep -qx "$JOB.2"; do
    sleep 10
done
sleep 30
PORT=8001 bash runs/qwen/launch_step.sh $JOB 8 logs/oracle/serve_c04-06_gemma_8001.out \
    /usr/bin/env PORT=8001 bash runs/qwen/serve_step.sh google/gemma-4-31b-it
until grep -q "vLLM healthy\|vLLM failed" logs/oracle/serve_c04-06_gemma_8001.out 2>/dev/null; do
    sleep 30
done
if grep -q "vLLM failed" logs/oracle/serve_c04-06_gemma_8001.out; then
    stamp "gemma server failed; not launching lane"
    exit 1
fi
stamp "gemma server healthy; launching oracle lane"
bash runs/qwen/launch_step.sh $JOB 16 logs/oracle/lane_gemma.out \
    /usr/bin/env VLLM_BASE_URL=http://localhost:8001/v1/ bash runs/oracle/lane.sh google/gemma-4-31b-it
stamp "handoff done"
