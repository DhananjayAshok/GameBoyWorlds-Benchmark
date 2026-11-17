#!/usr/bin/env bash
# Reports the practice stage: judged success vs the environment-verified benchmark rate,
# per-task judged success, clean-filter accept rates, and a judge-audit image pack. Writes
# <results_dir>/debug/<game>/practice/report.md.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array DEBUG_MODEL_ESSENTIALS REQUIRED_ARGS
populate_dict DEBUG_MODEL_DEFAULTS ARGS

ARGS["leg"]="both"
ARGS["n_audit"]=40
ARGS["n_frames"]=8
ARGS["seed"]=0

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

python debug.py $group_flags practice \
    --model_name "${ARGS["model_name"]}" \
    --leg "${ARGS["leg"]}" \
    --n_audit "${ARGS["n_audit"]}" \
    --n_frames "${ARGS["n_frames"]}" \
    --seed "${ARGS["seed"]}" || exit 1
