#!/usr/bin/env bash
# Annotates a trajectory with step-by-step guidance via infer_guidance, then
# immediately runs practice_tasks using the produced guidance file.
# guidance_path is derived as {trajectory_path}_guidance.json.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array GUIDANCE_AND_PRACTICE_ESSENTIALS REQUIRED_ARGS
populate_dict GUIDANCE_AND_PRACTICE_DEFAULTS ARGS

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

if [[ "${ARGS["practice_only"]}" == "true" || "${ARGS["practice_only"]}" == "yes" || "${ARGS["practice_only"]}" == "y" || "${ARGS["practice_only"]}" == "t" ]]; then
    ARGS["guidance_path"]="${ARGS["trajectory_path"]}_guidance.json"
    if [[ ! -f "${ARGS["guidance_path"]}" ]]; then
        echo "Error: guidance file not found at ${ARGS["guidance_path"]}. Run without --practice_only first."
        exit 1
    fi
else
    guidance_flags=$(args_to_flags_subset ARGS INFER_GUIDANCE_ARG_KEYS)
    bash scripts/vlm/infer_guidance.sh $guidance_flags || exit 1
    ARGS["guidance_path"]="${ARGS["trajectory_path"]}_guidance.json"
fi

if [[ "${ARGS["guidance_only"]}" == "true" || "${ARGS["guidance_only"]}" == "yes" || "${ARGS["guidance_only"]}" == "y" || "${ARGS["guidance_only"]}" == "t" ]]; then
    echo "guidance_only=true: skipping practice. guidance_path=${ARGS["guidance_path"]}"
else
    practice_flags=$(args_to_flags_subset ARGS PRACTICE_TASKS_ARG_KEYS)
    bash scripts/vlm/practice_tasks.sh $practice_flags || exit 1

    # practice_tasks writes to dirname(guidance_path)/practice_{executor}.
    ARGS["practice_path"]="$(dirname "${ARGS["guidance_path"]}")/practice_${ARGS["executor"]}"

    # create_dataset requires clean_practice's artifacts, so it only runs here.
    if [[ "${ARGS["do_clean"]}" == "true" || "${ARGS["do_clean"]}" == "yes" || "${ARGS["do_clean"]}" == "y" || "${ARGS["do_clean"]}" == "t" ]]; then
        clean_flags=$(args_to_flags_subset ARGS CLEAN_PRACTICE_ARG_KEYS)
        bash scripts/vlm/clean_practice.sh $clean_flags || exit 1

        create_dataset_flags=$(args_to_flags_subset ARGS CREATE_DATASET_ARG_KEYS)
        bash scripts/vlm/create_dataset.sh $create_dataset_flags || exit 1
    else
        echo "do_clean=false: skipping clean_practice and create_dataset."
    fi
fi
