#!/usr/bin/env bash
# Data collection followed by info-document construction, for one benchmark game:
# curiosity_and_zeroshot_all.sh once per train game of --game, then create_all_info_docs.sh
# once for --game itself.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array COLLECT_AND_INFO_ALL_ESSENTIALS REQUIRED_ARGS
populate_dict COLLECT_AND_INFO_ALL_DEFAULTS ARGS

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

mapfile -t train_games < <(path_of train_games --game "$game")
if [[ ${#train_games[@]} -eq 0 ]]; then
    echo "Error: no train games for $game (path_of train_games returned nothing)."
    exit 1
fi
echo "Train games for $game: ${train_games[*]}"

collected=""
failed=""
for train_game in "${train_games[@]}"; do
    echo ""
    echo "############################################################"
    echo "# $train_game  —  curiosity + zeroshot collection"
    echo "############################################################"

    ARGS["game"]="$train_game"
    flags=$(args_to_flags_subset ARGS PROPOSE_AND_ATTEMPT_ARG_KEYS)

    if bash scripts/pipeline/curiosity_and_zeroshot_all.sh $flags; then
        collected+="${collected:+ }$train_game"
    else
        echo "WARNING: collection failed for $train_game; continuing."
        failed+="${failed:+ }$train_game"
    fi
done
ARGS["game"]="$game"

echo ""
echo "############################################################"
echo "# $game  —  collection summary"
echo "############################################################"
echo "  collected: ${collected:-<none>}"
echo "  failed:    ${failed:-<none>}"

if [[ -z "$collected" ]]; then
    echo "Error: no train game collected for $game; nothing to build documents from."
    exit 1
fi

echo ""
echo "############################################################"
echo "# $game  —  create all info docs"
echo "############################################################"

# Swap in the info leg's own token budget: both legs spell the flag --max_new_tokens.
ARGS["max_new_tokens"]="${ARGS["info_max_new_tokens"]}"
flags=$(args_to_flags_subset ARGS CREATE_ALL_INFO_DOCS_ARG_KEYS)
bash scripts/vlm/create_all_info_docs.sh $flags || exit 1
