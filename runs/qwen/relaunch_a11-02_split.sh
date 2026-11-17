#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
JOB=6787492
L=logs/qwen
M8=Qwen/Qwen3-VL-8B-Instruct
CPUS=$(squeue -h -j $JOB -o "%C")

steps=$(squeue -h -s -j $JOB -o "%i" | grep -v "batch\|extern")
for s in $steps; do stamp "scancel $s"; scancel "$s"; done
for s in $steps; do
    while squeue -h -s -j $JOB -o "%i" | grep -qx "$s"; do sleep 10; done
done
sleep 30

stamp "starting 2B on GPU0:8001 and 8B on GPU1-3:8000"
bash runs/qwen/launch_step.sh $JOB 6 $L/serve_a11-02_2b_gpu0.out \
    /usr/bin/env PORT=8001 CUDA_VISIBLE_DEVICES=0 bash runs/qwen/serve_step.sh Qwen/Qwen3-VL-2B-Instruct -tp 1 --max-model-len 131072
bash runs/qwen/launch_step.sh $JOB 18 $L/serve_a11-02_8b_gpu123.out \
    /usr/bin/env PORT=8000 CUDA_VISIBLE_DEVICES=1,2,3 bash runs/qwen/serve_step.sh $M8 -tp 1 --data-parallel-size 3 --max-model-len 131072

for log in $L/serve_a11-02_2b_gpu0.out $L/serve_a11-02_8b_gpu123.out; do
    until grep -q "vLLM healthy\|vLLM failed" "$log" 2>/dev/null; do sleep 30; done
    grep -q "vLLM failed" "$log" && { stamp "SERVER FAILED: $log"; exit 1; }
    stamp "healthy: $log"
done

bash runs/qwen/launch_step.sh $JOB 4 $L/lane_2b_gaps_r2.out \
    /usr/bin/env VLLM_BASE_URL=http://localhost:8001/v1/ bash runs/qwen/lane_2b_gaps.sh
for g in bomberman_quest legend_of_zelda_links_awakening pokemon_red; do
    bash runs/qwen/launch_step.sh $JOB 4 $L/zs_8b_$g.out bash runs/qwen/lane_zs.sh $M8 $g
done
stamp "split relaunch done"
