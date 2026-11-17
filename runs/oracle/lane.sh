#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
source setup/.venv/bin/activate
source configs/config.env
source scripts/core/utils.sh
export TMPDIR="${TMPDIR:-/scratch1/$USER/tmp}"
mkdir -p "$TMPDIR"

MODEL="${1:-google/gemma-4-31b-it}"
GAMES="${2:-bomberman_quest,bomberman_pocket,deja_vu_1,deja_vu_2,legend_of_zelda_links_awakening,legend_of_zelda_the_oracle_of_seasons,pokemon_red,pokemon_crystal,sword_of_hope_1,sword_of_hope_2}"
LOGDIR=logs/oracle/$(model_save_name "$MODEL")
mkdir -p "$LOGDIR"

stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
stamp "oracle lane start host=$(hostname -s) model=$MODEL base_url=${VLLM_BASE_URL:-default} games=$GAMES"

IFS=',' read -ra games <<< "$GAMES"
for g in "${games[@]}"; do
    (
        stamp "bench $g start"
        bash scripts/benchmark/run_benchmark.sh \
            --game "$g" --supervisor subgoal --executor oracle_world_model --controller_variant low_level \
            --executor_vlm_model "$MODEL" --executor_vlm_kind vllm \
            --max_steps 175 \
            > "$LOGDIR/bench_${g}.out" 2>&1
        stamp "bench $g exit=$?"
    ) &
done
wait
stamp "oracle lane done"
