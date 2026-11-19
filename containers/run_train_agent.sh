#!/usr/bin/env bash
# Run prime-agent in a train sandbox, developing a harness on its own until it finishes or gives up, /work/STOP
# exists, the spending cap is reached, or --time_limit passes. Running the same line again continues the same run.
#
#     bash containers/run_train_agent.sh --run_dir <dir> --model google/gemini-3.6-flash --max_cost 1000 \
#         [--harness_repo harnesses/strategist] [--push_url <git url> --push_prefix <branch prefix>]
#
# Its instructions are containers/train_agent/AGENTS.md. To stop it: touch <run_dir>/work/STOP.
# Exits 0 when the agent finished and the server confirms both games cleared, 3 when it gave up.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source containers/common.sh

RUN_DIR=""
MODEL=""
MAX_COST=1000
TIME_LIMIT=""
extra=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_dir) RUN_DIR="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --max_cost) MAX_COST="$2"; shift 2 ;;
        --time_limit) TIME_LIMIT="$2"; shift 2 ;;
        --harness_repo|--push_url|--push_prefix) extra+=("$1" "$2"); shift 2 ;;
        *) sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
    esac
done
[[ -n "$RUN_DIR" ]] || die "--run_dir is required"
[[ -n "$MODEL" ]] || die "--model is required"

args=(--kind train --run_dir "$RUN_DIR" --model "$MODEL" --max_cost "$MAX_COST" "${extra[@]}")
[[ -z "$TIME_LIMIT" ]] || args+=(--time_limit "$TIME_LIMIT")
exec bash containers/run_sandbox.sh "${args[@]}" -- sh /opt/train/agent_loop.sh
