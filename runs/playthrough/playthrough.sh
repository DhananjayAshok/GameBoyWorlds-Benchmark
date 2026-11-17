#!/usr/bin/env bash
# One playthrough, then its debug reports. Does not start a vLLM server.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }
source configs/config.env || { echo "Could not source configs/config.env"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=("name" "model")

ARGS["game"]="pokemon_red"
ARGS["init_state"]="initial"
ARGS["vlm_kind"]="openrouter"
ARGS["max_episodes"]=400
ARGS["supervisor_max_steps"]=10
ARGS["subgoal_every"]=5
ARGS["report_detail"]="strategist"
ARGS["controller_variant"]="state_wise"
ARGS["save_video"]=true
ARGS["debug_reports"]=true
ARGS["debug_artifact"]="all"
OPTIONAL_FLAGS=(override_goal resume_run
    strategist_vlm_model strategist_vlm_kind
    supervisor_vlm_model supervisor_vlm_kind
    executor_vlm_model executor_vlm_kind)
for flag in "${OPTIONAL_FLAGS[@]}"; do ARGS["$flag"]="none"; done

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

optional_flags=()
for flag in "${OPTIONAL_FLAGS[@]}"; do
    if [[ "${ARGS[$flag]}" != "none" ]]; then
        optional_flags+=("--$flag" "${ARGS[$flag]}")
    fi
done

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
    --save_video "${ARGS["save_video"]}" \
    "${optional_flags[@]}"
run_status=$?

# Reports are rendered even when the playthrough failed; this script exits with its status.
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
