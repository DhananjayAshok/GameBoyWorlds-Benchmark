#!/usr/bin/env bash
# run_benchmark.sh's info_subgoal_retrieval arm, with --info_docs resolved from the train games
# of --game rather than typed. --docs_model / --docs_executor / --docs_controller_variant /
# --docs_run_name / --docs_mode select which documents to read; all of them must exist.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array RUN_BENCHMARK_INFO_RETRIEVAL_ESSENTIALS REQUIRED_ARGS
populate_dict RUN_BENCHMARK_INFO_RETRIEVAL_DEFAULTS ARGS

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
docs_model="${ARGS["docs_model"]}"
docs_executor="${ARGS["docs_executor"]}"
docs_controller_variant="${ARGS["docs_controller_variant"]}"
docs_run_name="${ARGS["docs_run_name"]}"

wanted_sources=$(info_mode_sources "${ARGS["docs_mode"]}")
if [[ -z "$wanted_sources" ]]; then
    echo "Error: unknown --docs_mode '${ARGS["docs_mode"]}'. Choose one of: curiosity_only, zeroshot_only, both."
    exit 1
fi
echo "docs_mode '${ARGS["docs_mode"]}' -> sources: $wanted_sources"

# path_of's own exit only kills the subshell here, so an empty list is checked for below.
mapfile -t docs_games < <(path_of train_games --game "$game")
if [[ ${#docs_games[@]} -eq 0 ]]; then
    echo "Error: no train games for $game (path_of train_games returned nothing)."
    exit 1
fi
echo "Train games for $game: ${docs_games[*]}"

info_docs=""
missing=""
for docs_game in "${docs_games[@]}"; do
    for source in $wanted_sources; do
        doc=$(path_of info_doc --game "$docs_game" --model_name "$docs_model" \
                  --run_name "$docs_run_name" --executor "$docs_executor" \
                  --controller_variant "$docs_controller_variant" --source "$source")
        if [[ -f "$doc" ]]; then
            info_docs+="${info_docs:+,}$doc"
            echo "  $docs_game/$source -> $doc"
        else
            echo "  $docs_game/$source -> MISSING $doc"
            missing+="${missing:+ }$docs_game/$source"
        fi
    done
done

if [[ -n "$missing" ]]; then
    echo "Error: --docs_mode '${ARGS["docs_mode"]}' requires [$wanted_sources] for every train"
    echo "  game of $game, but these are not on disk: $missing"
    echo "  Looked under --docs_model $docs_model --docs_run_name $docs_run_name --docs_executor $docs_executor --docs_controller_variant $docs_controller_variant"
    echo "  A MISSING line above with insights.jsonl beside it means stage B has not run under"
    echo "  the current JSON format: scripts/vlm/create_all_info_docs.sh"
    echo "  To plan from only the vertical that exists, pass --docs_mode zeroshot_only or curiosity_only."
    exit 1
fi

ARGS["supervisor"]="info_subgoal_retrieval"
ARGS["info_docs"]="$info_docs"

# Default extra_name to docs_mode so runs differing only in docs_mode do not share a CSV.
if [[ "${ARGS["extra_name"]}" == "none" ]]; then
    ARGS["extra_name"]="${ARGS["docs_mode"]}"
    echo "extra_name defaulted to docs_mode '${ARGS["docs_mode"]}' (names the CSV and session dir)"
fi

# Subset against run_benchmark.sh's own key list, which drops the docs_* flags it does not accept.
arg_string=$(args_to_flags_subset ARGS RUN_BENCHMARK_ARG_KEYS)
bash scripts/benchmark/run_benchmark.sh $arg_string
