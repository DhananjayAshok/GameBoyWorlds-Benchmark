#!/usr/bin/env bash
# One long-horizon strategist playthrough. The invocation the slurm/strategist_*.sh files
# wrap, so the six of them stop each carrying their own copy.
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
ARGS["explore_buildings"]=false

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
buildings_flag=""
if [[ "${ARGS["explore_buildings"]}" == "true" || "${ARGS["explore_buildings"]}" == "yes" || "${ARGS["explore_buildings"]}" == "y" ]]; then
    buildings_flag="--explore_buildings"
fi

python run_strategist.py \
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
    $resume_flag $buildings_flag

echo "DONE: ${ARGS["name"]} on ${ARGS["game"]}"
