#!/usr/bin/env bash
# Runs an arbitrary command inside the llm-utils subdirectory using its own virtual
# environment, then switches back to the project environment. Used by train_vlm.sh.

if [ $# -eq 0 ]; then
    echo "Usage: bash llm-utils.sh <command_to_run>"
    exit 1
fi

to_run_command="$@"

currdir="$PWD"
source scripts/core/utils.sh || { echo "Could not source utils.sh"; exit 1; }
python configs/create_env_file.py || { echo "Could not create env file"; exit 1; }

source "llm-utils/setup/.venv/bin/activate" || { echo "Could not source llm-utils env"; exit 1; }

cd llm-utils || { echo "Could not cd into llm-utils"; exit 1; }

echo "Running command: $to_run_command"
eval "$to_run_command"


cd "$currdir" || { echo "Could not return to original directory"; exit 1; }
source "$env_dir/bin/activate" || { echo "Could not source project env"; exit 1; }