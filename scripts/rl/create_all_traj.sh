#!/usr/bin/env bash
# Batch wrapper that runs create_traj.sh for every init_state registered for a game in
# TRAIN_STATES, regenerating the state dictionary from GameBoyWorlds first.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }
python "$PROJECT_ROOT/python_funcs.py" task_dictionary || { echo "Could not regenerate train states"; exit 1; }
source scripts/core/all_train_states.sh

declare -A ARGS
REQUIRED_ARGS=()
populate_array CREATE_TRAJ_ESSENTIALS REQUIRED_ARGS
# init_state comes from the TRAIN_STATES loop below, not the CLI.
for i in "${!REQUIRED_ARGS[@]}"; do
    [[ "${REQUIRED_ARGS[$i]}" == "init_state" ]] && unset 'REQUIRED_ARGS[$i]'
done
REQUIRED_ARGS=("${REQUIRED_ARGS[@]}")
populate_dict CREATE_TRAJ_DEFAULTS ARGS


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

if [[ -z "${ARGS["run_name"]}" ]] || [[ "${ARGS["run_name"]}" == "default" ]] || [[ "${ARGS["run_name"]}" == "iterative" ]]; then
    echo "Error: --run_name is required and cannot be 'default' or 'iterative'."
    usage
fi

game="${ARGS["game"]}"
IFS=',' read -ra init_states_arr <<< "${TRAIN_STATES[$game]}"
for init_state in "${init_states_arr[@]}"; do
    ARGS["init_state"]=$init_state
    arg_string=$(args_to_flags_subset ARGS CREATE_TRAJ_ARG_KEYS)
    bash scripts/rl/create_traj.sh $arg_string
done