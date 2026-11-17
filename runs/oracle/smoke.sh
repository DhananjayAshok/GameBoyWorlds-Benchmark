#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
source setup/.venv/bin/activate
source configs/config.env
export TMPDIR="${TMPDIR:-/scratch1/$USER/tmp}"
mkdir -p logs/oracle
for g in pokemon_red legend_of_zelda_links_awakening; do
    bash scripts/benchmark/run_benchmark.sh \
        --game "$g" --supervisor subgoal --executor oracle_world_model --controller_variant low_level \
        --executor_vlm_model google/gemini-3.6-flash --executor_vlm_kind openrouter \
        --n_tasks 2 --max_steps 20 --executor_max_new_tokens 2000 --supervisor_max_new_tokens 2000 \
        > logs/oracle/smoke_$g.out 2>&1
    echo "$g exit=$?"
done
