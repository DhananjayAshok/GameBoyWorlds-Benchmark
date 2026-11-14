#!/usr/bin/env bash
# One long-horizon playthrough. The invocation the slurm/playthrough_*.sh files wrap, so
# they stop each carrying their own copy.
#
# Does NOT start a vLLM server. For --vlm_kind vllm, bring one up first.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }
source configs/config.env || { echo "Could not source configs/config.env"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=("name" "model")

ARGS["game"]="pokemon_red"
ARGS["init_state"]="initial"
ARGS["vlm_kind"]="openrouter"
ARGS["max_episodes"]=75
ARGS["supervisor_max_steps"]=10
ARGS["subgoal_every"]=5
ARGS["report_detail"]="strategist"
ARGS["controller_variant"]="state_wise"
ARGS["resume"]=true
# Render the debug view/diff reports once the playthrough returns. On by default: they are
# cheap markdown over artifacts already on disk, and every per-run slurm job wanted them.
ARGS["debug_reports"]=true
# Which artifact those reports cover. "all" passes no --artifact, so debug.py reports on
# every artifact the run saved.
ARGS["debug_artifact"]="all"

# --- Argument parsing (copy verbatim) ---
ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")
USAGE_STR="Usage: $0"
for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done
for opt in "${!ARGS[@]}"; do
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        if [[ -z "${ARGS[$opt]}" ]]; then
            echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"; exit 1
        fi
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done
function usage() { echo "$USAGE_STR"; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        --*)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then VALID=true; break; fi
            done
            if [ "$VALID" = false ]; then echo "Error: Unknown flag --$FLAG"; usage; fi
            ARGS["$FLAG"]="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then echo "Error: --$req is required."; FAILED=true; fi
done
if [ "$FAILED" = true ]; then usage; fi
# --- End argument parsing ---

echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

# --resume/--fresh is a click flag pair, not a --flag value option.
if [[ "${ARGS["resume"]}" == "true" || "${ARGS["resume"]}" == "yes" || "${ARGS["resume"]}" == "y" ]]; then
    resume_flag="--resume"
else
    resume_flag="--fresh"
fi

python run_playthrough.py \
    --game "${ARGS["game"]}" \
    --init_state "${ARGS["init_state"]}" \
    --name "${ARGS["name"]}" \
    --model "${ARGS["model"]}" \
    --vlm_kind "${ARGS["vlm_kind"]}" \
    --max_episodes "${ARGS["max_episodes"]}" \
    --supervisor_max_steps "${ARGS["supervisor_max_steps"]}" \
    --subgoal_every "${ARGS["subgoal_every"]}" \
    --report_detail "${ARGS["report_detail"]}" \
    --controller_variant "${ARGS["controller_variant"]}" \
    $resume_flag
run_status=$?

# Reports are rendered even when the playthrough exited non-zero: a run killed by the wall
# clock has written every episode up to that point, and that is exactly when the reports are
# wanted. The run's own status is what this script exits with.
if [[ "${ARGS["debug_reports"]}" == "true" || "${ARGS["debug_reports"]}" == "yes" ]]; then
    artifact_flag=()
    if [[ "${ARGS["debug_artifact"]}" != "all" ]]; then
        artifact_flag=(--artifact "${ARGS["debug_artifact"]}")
    fi
    python debug.py --game "${ARGS["game"]}" playthrough view \
        --name "${ARGS["name"]}" "${artifact_flag[@]}"
    python debug.py --game "${ARGS["game"]}" playthrough diff \
        --name "${ARGS["name"]}" "${artifact_flag[@]}"
fi

echo "DONE: ${ARGS["name"]} on ${ARGS["game"]}"
exit $run_status
