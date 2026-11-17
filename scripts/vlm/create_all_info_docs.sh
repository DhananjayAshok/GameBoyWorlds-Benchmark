#!/usr/bin/env bash

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array CREATE_ALL_INFO_DOCS_ESSENTIALS REQUIRED_ARGS
populate_dict CREATE_ALL_INFO_DOCS_DEFAULTS ARGS

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
mode="${ARGS["mode"]}"

sources=$(info_mode_sources "$mode")
if [[ -z "$sources" ]]; then
    echo "Error: unknown --mode '$mode'. Choose one of: curiosity_only, zeroshot_only, both."
    exit 1
fi

mapfile -t docs_games < <(path_of train_games --game "$game")
if [[ ${#docs_games[@]} -eq 0 ]]; then
    echo "Error: no train games for $game (path_of train_games returned nothing)."
    exit 1
fi

echo "Train games for $game: ${docs_games[*]}"
echo "Mode '$mode' -> sources: $sources"

built=""
failed=""
for docs_game in "${docs_games[@]}"; do
    for source_name in $sources; do
        echo ""
        echo "############################################################"
        echo "# $docs_game  —  info document: $source_name"
        echo "############################################################"

        ARGS["game"]="$docs_game"
        ARGS["source"]="$source_name"
        flags=$(args_to_flags_subset ARGS CREATE_INFO_DOC_ARG_KEYS)

        if bash scripts/vlm/create_info_doc.sh $flags; then
            built+="${built:+ }$docs_game/$source_name"
        else
            echo "WARNING: info document failed for $docs_game/$source_name; continuing."
            failed+="${failed:+ }$docs_game/$source_name"
        fi
    done
done
ARGS["game"]="$game"

echo ""
echo "############################################################"
echo "# $game  —  summary"
echo "############################################################"
echo "  built:  ${built:-<none>}"
echo "  failed: ${failed:-<none>}"

if [[ -z "$built" ]]; then
    echo "Error: no info documents built for $game (train games: ${docs_games[*]})."
    exit 1
fi
