#!/usr/bin/env bash
# Summarises the curiosity/grouping stage from the saved grouped-trajectory pickles: group
# and trajectory counts, reward distributions, and per-group frame strips. Writes
# <results_dir>/debug/<game>/curiosity/report.md. Needs no model.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array DEBUG_ESSENTIALS REQUIRED_ARGS
populate_dict DEBUG_DEFAULTS ARGS

ARGS["init_state"]="all_states"
ARGS["n_frames"]=5
ARGS["max_groups"]=0
ARGS["max_traj_per_group"]=3
ARGS["max_gb"]=8.0

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

group_flags=$(debug_group_flags ARGS)

python debug.py $group_flags curiosity \
    --init_state "${ARGS["init_state"]}" \
    --n_frames "${ARGS["n_frames"]}" \
    --max_groups "${ARGS["max_groups"]}" \
    --max_traj_per_group "${ARGS["max_traj_per_group"]}" \
    --max_gb "${ARGS["max_gb"]}" || exit 1
