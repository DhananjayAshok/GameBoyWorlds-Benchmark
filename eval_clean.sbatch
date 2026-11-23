#!/bin/bash
#SBATCH --account=jonmay_231
#SBATCH --partition=isi
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/project2/jessetho_1732/ashokd/GameBoyRL/env_server/runs/eval_logs/%x_%j.out

module purge
module load apptainer/1.5.5

GAME="$1"
NAME="eval_clean_${GAME#pokemon_}"
MODEL=google/gemini-4-argon

export TMPDIR=/scratch1/ashokd/tmp
mkdir -p "$TMPDIR"
export PYTHONPATH=/home1/ashokd/projects/GameBoyRL_main/GameBoyWorlds/src
export SANDBOX_SIF=/project2/jessetho_1732/ashokd/GameBoyRL/containers/sandbox_repl.sif

cd /home1/ashokd/projects/GameBoyRL_main
bash containers/run_sandbox.sh --kind playthrough --task_id "gb.$GAME.playthrough" \
    --run_dir "/project2/jessetho_1732/ashokd/GameBoyRL/env_server/runs/$NAME" \
    --model "$MODEL" --max_cost 150 --harness --stop_after_no_progress 1000 -- \
    /opt/harness/bin/python /work/harness/run_playthrough.py --remote --name "$NAME" --model "$MODEL" \
    --vlm_kind gateway
