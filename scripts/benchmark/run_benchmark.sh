#!/usr/bin/env bash
# Plugs into run_benchmark.py

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array RUN_BENCHMARK_ESSENTIALS REQUIRED_ARGS
populate_dict RUN_BENCHMARK_DEFAULTS ARGS

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

echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

optional_flags=""
if [[ "${ARGS["supervisor_vlm_model"]}" != "none" ]]; then
    optional_flags+=" --supervisor_vlm_model ${ARGS["supervisor_vlm_model"]}"
fi
if [[ "${ARGS["supervisor_vlm_kind"]}" != "none" ]]; then
    optional_flags+=" --supervisor_vlm_kind ${ARGS["supervisor_vlm_kind"]}"
fi
if [[ "${ARGS["n_tasks"]}" != "none" ]]; then
    optional_flags+=" --n_tasks ${ARGS["n_tasks"]}"
fi
if [[ "${ARGS["verbose"]}" == "true" ]]; then
    optional_flags+=" --verbose"
fi
if [[ "${ARGS["regenerate"]}" == "true" ]]; then
    optional_flags+=" --regenerate"
fi
if [[ "${ARGS["extra_name"]}" != "none" ]]; then
    optional_flags+=" --extra_name ${ARGS["extra_name"]}"
fi
# Only the world_model executor reads these.
if [[ "${ARGS["world_model_run_name"]}" != "none" ]]; then
    optional_flags+=" --world_model_run_name ${ARGS["world_model_run_name"]}"
fi
if [[ "${ARGS["world_model_game"]}" != "none" ]]; then
    optional_flags+=" --world_model_game ${ARGS["world_model_game"]}"
fi

group_flags="--game ${ARGS["game"]}"
group_flags+=" --controller_variant ${ARGS["controller_variant"]}"
group_flags+=" --executor ${ARGS["executor"]}"
group_flags+=" --executor_vlm_model ${ARGS["executor_vlm_model"]}"
group_flags+=" --executor_vlm_kind ${ARGS["executor_vlm_kind"]}"
group_flags+=" --supervisor_max_new_tokens ${ARGS["supervisor_max_new_tokens"]}"
group_flags+=" --save_video ${ARGS["save_video"]}"
group_flags+=" --max_steps ${ARGS["max_steps"]}"
group_flags+="$optional_flags"

case "${ARGS["supervisor"]}" in
    dummy)
        subcommand="baseline"
        arm_flags="" ;;
    revision)
        subcommand="revision"
        arm_flags="--max_leg_steps ${ARGS["max_leg_steps"]}"
        arm_flags+=" --max_frames_per_slice ${ARGS["max_frames_per_slice"]}"
        arm_flags+=" --executor_max_new_tokens ${ARGS["executor_max_new_tokens"]}" ;;
    subgoal)
        subcommand="subgoal"
        arm_flags="--max_leg_steps ${ARGS["max_leg_steps"]}"
        arm_flags+=" --max_attempts_per_step ${ARGS["max_attempts_per_step"]}"
        arm_flags+=" --max_replans ${ARGS["max_replans"]}"
        arm_flags+=" --max_frames_per_slice ${ARGS["max_frames_per_slice"]}"
        arm_flags+=" --executor_max_new_tokens ${ARGS["executor_max_new_tokens"]}" ;;
    info_subgoal_retrieval)
        subcommand="info_subgoal_retrieval"
        arm_flags="--max_concurrency ${ARGS["max_concurrency"]}"
        arm_flags+=" --max_leg_steps ${ARGS["max_leg_steps"]}"
        arm_flags+=" --max_attempts_per_step ${ARGS["max_attempts_per_step"]}"
        arm_flags+=" --max_replans ${ARGS["max_replans"]}"
        arm_flags+=" --max_frames_per_slice ${ARGS["max_frames_per_slice"]}"
        arm_flags+=" --executor_max_new_tokens ${ARGS["executor_max_new_tokens"]}"
        if [[ "${ARGS["info_docs"]}" != "none" ]]; then
            arm_flags+=" --info_docs ${ARGS["info_docs"]}"
        fi ;;
    info_subgoal_parametric)
        subcommand="info_subgoal_parametric"
        arm_flags="--parametric_categories ${ARGS["parametric_categories"]}"
        arm_flags+=" --max_concurrency ${ARGS["max_concurrency"]}"
        arm_flags+=" --max_leg_steps ${ARGS["max_leg_steps"]}"
        arm_flags+=" --max_attempts_per_step ${ARGS["max_attempts_per_step"]}"
        arm_flags+=" --max_replans ${ARGS["max_replans"]}"
        arm_flags+=" --max_frames_per_slice ${ARGS["max_frames_per_slice"]}"
        arm_flags+=" --executor_max_new_tokens ${ARGS["executor_max_new_tokens"]}" ;;
    *)
        echo "Error: unknown --supervisor '${ARGS["supervisor"]}' (dummy, revision, subgoal, info_subgoal_retrieval, info_subgoal_parametric)"
        exit 1 ;;
esac

command="python run_benchmark.py $group_flags $subcommand $arm_flags"

echo "$command"
$command
