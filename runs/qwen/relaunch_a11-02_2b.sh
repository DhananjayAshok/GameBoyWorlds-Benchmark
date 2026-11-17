#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
bash runs/qwen/restart_server.sh 6787492 8 logs/qwen/serve_a11-02_2b.out "" \
    Qwen/Qwen3-VL-2B-Instruct -tp 1 --data-parallel-size 4 --max-model-len 131072 || exit 1
bash runs/qwen/launch_step.sh 6787492 8 logs/qwen/lane_2b_gaps.out bash runs/qwen/lane_2b_gaps.sh
