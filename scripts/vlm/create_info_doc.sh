#!/usr/bin/env bash

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array CREATE_INFO_DOC_ESSENTIALS REQUIRED_ARGS
populate_dict CREATE_INFO_DOC_DEFAULTS ARGS

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

game="${ARGS["game"]}"
source_name="${ARGS["source"]}"

case "$source_name" in
    zeroshot|curiosity) ;;
    *) echo "Error: unknown --source '$source_name'. Choose one of: zeroshot, curiosity."
       exit 1 ;;
esac

stem=$(path_of info_source_stem --game "$game" --model_name "${ARGS["model_name"]}" \
               --run_name "${ARGS["run_name"]}" --executor "${ARGS["executor"]}" \
               --controller_variant "${ARGS["controller_variant"]}" --source "$source_name")

info_dir=$(path_of source_info_dir --game "$game" --model_name "${ARGS["model_name"]}" \
                   --run_name "${ARGS["run_name"]}" --executor "${ARGS["executor"]}" \
                   --controller_variant "${ARGS["controller_variant"]}" --source "$source_name")

echo "$game/$source_name: $stem -> $info_dir"

ARGS["trajectory_path"]="$stem"
ARGS["stage"]="all"

flags=$(args_to_flags_subset ARGS BUILD_INFO_ARG_KEYS)
bash scripts/vlm/build_info.sh $flags || exit 1

echo "$game/$source_name -> $info_dir"
