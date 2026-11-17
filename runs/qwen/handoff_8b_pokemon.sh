#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
source configs/config.env
D=$storage_dir/proposed_tasks/pokemon_red/qwen3-vl-8b-instruct/curiosity/my_run
until [[ -f $D/trajectory_annotation.json && -f $D/trajectory_annotation.pkl && ! -f $D/trajectory_annotation_checkpoint.json ]]; do
    sleep 30
done
sleep 30
echo "[$(date '+%m-%d %H:%M:%S')] annotation landed; cancelling 6787478.3"
scancel 6787478.3
sleep 20
bash runs/qwen/launch_step.sh 6787478 8 logs/qwen/cur_8b_pokemon.out bash runs/qwen/lane_cur.sh Qwen/Qwen3-VL-8B-Instruct pokemon_red pokemon_red,pokemon_crystal
