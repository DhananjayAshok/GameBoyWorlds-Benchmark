# Core shared library sourced by every other script in this project.
# Loads the project config and virtual environment, then defines the default argument
# dictionaries, the *_ARG_KEYS arrays, and the argument parsing/forwarding helpers.

source configs/config.env || { echo "configs/config.env not found"; exit 1; }
source setup/.venv/bin/activate || { echo "Virtual environment not found."; exit 1; }
PROJECT_ROOT=$(pwd)

# Converts an associative array into a flat --key value string; empty values become --key none.
function args_to_flags() {
    local -n _dict="$1"
    local result=""
    for key in "${!_dict[@]}"; do
        local val="${_dict[$key]}"
        if [[ -z "$val" ]]; then
            val="none"
        fi
        result+="--${key} ${val} "
    done
    echo "${result% }"  # trim trailing space
}

function get_string_from_args() {
    local string_kind="$1"
    shift
    local flags=$(args_to_flags "$1")
    python "$PROJECT_ROOT/python_funcs.py" strings "$string_kind" $flags
}



# Like args_to_flags, but only emits flags for the specified keys; absent keys are skipped.
function args_to_flags_subset() {
    local -n _dict="$1"
    local -n _keys="$2"
    local result=""
    for key in "${_keys[@]}"; do
        if [[ -v _dict["$key"] ]]; then
            local val="${_dict[$key]}"
            if [[ -z "$val" ]]; then val="none"; fi
            result+="--${key} ${val} "
        fi
    done
    echo "${result% }"
}

function populate_dict(){
    local -n _source_dict="$1"
    local -n _target_dict="$2"
    for key in "${!_source_dict[@]}"; do
        _target_dict["$key"]="${_source_dict[$key]}"
    done
}

function populate_array(){
    local -n _source_arr="$1"
    local -n _target_arr="$2"
    _target_arr+=("${_source_arr[@]}")
}

function populate_dict_subset(){
    local -n _source_dict="$1"
    local -n _target_dict="$2"
    local -n _subset_keys="$3"
    for key in "${_subset_keys[@]}"; do
        if [[ -v _source_dict["$key"] ]]; then
            _target_dict["$key"]="${_source_dict[$key]}"
        fi
    done
}

# The basename the pipeline keys paths on: google/Gemma-4-31B-it -> gemma-4-31b-it.
# Must agree with model_save_name() in python_scripts/paths.py.
# Names directories and files ONLY. Never pass this to a backend as a model identifier.
function model_save_name() {
    local base="${1##*/}"
    echo "${base,,}"
}

# Ask python_funcs.py for a path. THE way Bash gets any path.
# Wrapped rather than called directly because $(...) swallows a non-zero exit.
function path_of() {
    local out
    if ! out=$(python "$PROJECT_ROOT/python_funcs.py" "$@"); then
        echo "path_of: lookup failed: python_funcs.py $*" >&2
        exit 1
    fi
    if [[ -z "$out" ]]; then
        echo "path_of: lookup returned empty: python_funcs.py $*" >&2
        exit 1
    fi
    echo "$out"
}

# Same, but an empty result is a legitimate answer rather than an error.
function path_of_allow_empty() {
    local out
    if ! out=$(python "$PROJECT_ROOT/python_funcs.py" "$@"); then
        echo "path_of_allow_empty: lookup failed: python_funcs.py $*" >&2
        exit 1
    fi
    echo "$out"
}


ESSENTIAL_ARGS=("game") # should be game later. 
declare -A ALL_DEFAULTS=( # make this empty later
)

ENV_ESSENTIALS=("init_state")
declare -A ENV_DEFAULTS=(
    ["env"]="default"
    ["controller"]="low_level"
    ["max_steps"]=30
)


declare -A ALGORITHM_DEFAULTS=(
    ["timesteps"]=100000
    ["algorithm"]="ppo"
    ["gamma"]="0.99"
    ["seed"]=1
)

declare -A CURIOUSITY_DEFAULTS=(
    ["curiosity_module"]="combinationbuffer"
    ["ocr_alpha"]=0.0
    ["buffer_load_path"]="none"
    ["similarity_metric"]="cosine"    
    ["observation_embedder"]="random_patch"
    ["embedder_load_path"]="none"    
)

declare -A SAVE_DEFAULTS=(
    ["buffer_save_path"]="none"
    ["replay_buffer_save_folder"]="none"
    ["model_dir"]="none"
    ["overwrite_model"]=false
    ["log_folder"]="none"
    ["capture_video"]=false
)

declare -A TRAINING_DEFAULTS
TRAINING_ESSENTIALS=()
populate_dict ALL_DEFAULTS TRAINING_DEFAULTS
populate_dict ENV_DEFAULTS TRAINING_DEFAULTS
populate_dict ALGORITHM_DEFAULTS TRAINING_DEFAULTS
populate_dict CURIOUSITY_DEFAULTS TRAINING_DEFAULTS
populate_dict SAVE_DEFAULTS TRAINING_DEFAULTS
populate_array ESSENTIAL_ARGS TRAINING_ESSENTIALS
populate_array ENV_ESSENTIALS TRAINING_ESSENTIALS

TRAINING_ARG_KEYS=("${TRAINING_ESSENTIALS[@]}" "${!TRAINING_DEFAULTS[@]}")

declare -A EVALUATION_DEFAULTS
EVALUATION_ESSENTIALS=()
populate_dict ALL_DEFAULTS EVALUATION_DEFAULTS
populate_dict ENV_DEFAULTS EVALUATION_DEFAULTS
populate_dict SAVE_DEFAULTS EVALUATION_DEFAULTS
populate_dict CURIOUSITY_DEFAULTS EVALUATION_DEFAULTS
populate_array ESSENTIAL_ARGS EVALUATION_ESSENTIALS
populate_array ENV_ESSENTIALS EVALUATION_ESSENTIALS

EVALUATION_ESSENTIALS+=("algorithm" "exp_name")

EVALUATION_ARG_KEYS=("${EVALUATION_ESSENTIALS[@]}" "${!EVALUATION_DEFAULTS[@]}")

declare -A WORLD_MODEL_DEFAULTS
WORLD_MODEL_ESSENTIALS=("latest_replay_buffer_folder" "game")
populate_dict ALL_DEFAULTS WORLD_MODEL_DEFAULTS
populate_array ESSENTIAL_ARGS WORLD_MODEL_ESSENTIALS
SAME_AS_TRAINING=("observation_embedder" "embedder_load_path" "buffer_save_path" "buffer_load_path" "controller")
populate_dict_subset TRAINING_DEFAULTS WORLD_MODEL_DEFAULTS SAME_AS_TRAINING

WORLD_MODEL_ARG_KEYS=("${WORLD_MODEL_ESSENTIALS[@]}" "${!WORLD_MODEL_DEFAULTS[@]}")

declare -A SWEEP_DEFAULTS
SWEEP_ESSENTIALS=()
populate_array ESSENTIAL_ARGS SWEEP_ESSENTIALS
populate_array ENV_ESSENTIALS SWEEP_ESSENTIALS
populate_dict TRAINING_DEFAULTS SWEEP_DEFAULTS
SWEEP_DEFAULTS["best_k"]=6
SWEEP_DEFAULTS["clear_loser_replay_buffer"]=true

SWEEP_ARG_KEYS=("${SWEEP_ESSENTIALS[@]}" "${!SWEEP_DEFAULTS[@]}")

ITERATIVE_TRAINING_ESSENTIALS=()
declare -A ITERATIVE_TRAINING_DEFAULTS
populate_array ESSENTIAL_ARGS ITERATIVE_TRAINING_ESSENTIALS
populate_array ENV_ESSENTIALS ITERATIVE_TRAINING_ESSENTIALS
populate_dict ALL_DEFAULTS ITERATIVE_TRAINING_DEFAULTS
populate_dict SWEEP_DEFAULTS ITERATIVE_TRAINING_DEFAULTS
ITERATIVE_TRAINING_DEFAULTS["n_agents"]=3
ITERATIVE_TRAINING_DEFAULTS["sweep"]=false
ITERATIVE_TRAINING_DEFAULTS["combination_buffer_sweep"]=true
ITERATIVE_TRAINING_DEFAULTS["init_state_group"]="none"
ITERATIVE_TRAINING_DEFAULTS["run_name"]="iterative"

ITERATIVE_TRAINING_ARG_KEYS=("${ITERATIVE_TRAINING_ESSENTIALS[@]}" "${!ITERATIVE_TRAINING_DEFAULTS[@]}")


CREATE_TRAJ_ESSENTIALS=()
declare -A CREATE_TRAJ_DEFAULTS
populate_array ITERATIVE_TRAINING_ESSENTIALS CREATE_TRAJ_ESSENTIALS
populate_dict ALL_DEFAULTS CREATE_TRAJ_DEFAULTS
populate_dict ITERATIVE_TRAINING_DEFAULTS CREATE_TRAJ_DEFAULTS
CREATE_TRAJ_DEFAULTS["skip_training"]=false
CREATE_TRAJ_DEFAULTS["z_min"]=6.0
CREATE_TRAJ_DEFAULTS["overwrite"]=false
CREATE_TRAJ_DEFAULTS["keep_buffers"]=false
# When false, the run stops after training and the buffers are kept regardless of --keep_buffers.
CREATE_TRAJ_DEFAULTS["group_trajectories"]=true

CREATE_TRAJ_ARG_KEYS=("${CREATE_TRAJ_ESSENTIALS[@]}" "${!CREATE_TRAJ_DEFAULTS[@]}")

VLM_ESSENTIALS=("model_name" "vlm_kind")
populate_array ESSENTIAL_ARGS VLM_ESSENTIALS
declare -A VLM_DEFAULTS=(
    ["overwrite"]=false
    ["verbose"]=false
    ["max_new_tokens"]=1000
)
VLM_ARG_KEYS=("${VLM_ESSENTIALS[@]}" "${!VLM_DEFAULTS[@]}")

PROPOSE_ZEROSHOT_ESSENTIALS=("init_states")
populate_array VLM_ESSENTIALS PROPOSE_ZEROSHOT_ESSENTIALS
declare -A PROPOSE_ZEROSHOT_DEFAULTS
populate_dict VLM_DEFAULTS PROPOSE_ZEROSHOT_DEFAULTS

PROPOSE_ZEROSHOT_ARG_KEYS=("${PROPOSE_ZEROSHOT_ESSENTIALS[@]}" "${!PROPOSE_ZEROSHOT_DEFAULTS[@]}")

ATTEMPT_TASKS_ESSENTIALS=("tasks_path")
populate_array VLM_ESSENTIALS ATTEMPT_TASKS_ESSENTIALS
declare -A ATTEMPT_TASKS_DEFAULTS
populate_dict VLM_DEFAULTS ATTEMPT_TASKS_DEFAULTS
ATTEMPT_TASKS_DEFAULTS["executor"]="single_actions"
ATTEMPT_TASKS_DEFAULTS["max_steps"]=50
ATTEMPT_TASKS_DEFAULTS["lookback"]=8
ATTEMPT_TASKS_DEFAULTS["controller_variant"]="low_level"
ATTEMPT_TASKS_DEFAULTS["max_attempts"]=5
ATTEMPT_TASKS_DEFAULTS["checker_max_new_tokens"]=1000

ATTEMPT_TASKS_ARG_KEYS=("${ATTEMPT_TASKS_ESSENTIALS[@]}" "${!ATTEMPT_TASKS_DEFAULTS[@]}")

INFER_TASKS_ESSENTIALS=("trajectory_path")
populate_array VLM_ESSENTIALS INFER_TASKS_ESSENTIALS
declare -A INFER_TASKS_DEFAULTS
populate_dict VLM_DEFAULTS INFER_TASKS_DEFAULTS
INFER_TASKS_DEFAULTS["run_name"]="all"
INFER_TASKS_DEFAULTS["lookback"]=8
INFER_TASKS_DEFAULTS["max_trajectories_per_group"]=3
INFER_TASKS_DEFAULTS["describe_pairs"]=false

INFER_TASKS_ARG_KEYS=("${INFER_TASKS_ESSENTIALS[@]}" "${!INFER_TASKS_DEFAULTS[@]}")


INFER_GUIDANCE_ESSENTIALS=("trajectory_path")
populate_array VLM_ESSENTIALS INFER_GUIDANCE_ESSENTIALS
declare -A INFER_GUIDANCE_DEFAULTS
populate_dict VLM_DEFAULTS INFER_GUIDANCE_DEFAULTS
INFER_GUIDANCE_DEFAULTS["max_obs_at_once"]=8
INFER_GUIDANCE_DEFAULTS["max_concurrency"]=16

INFER_GUIDANCE_ARG_KEYS=("${INFER_GUIDANCE_ESSENTIALS[@]}" "${!INFER_GUIDANCE_DEFAULTS[@]}")

PRACTICE_TASKS_ESSENTIALS=("guidance_path")
populate_array VLM_ESSENTIALS PRACTICE_TASKS_ESSENTIALS
declare -A PRACTICE_TASKS_DEFAULTS
populate_dict VLM_DEFAULTS PRACTICE_TASKS_DEFAULTS
PRACTICE_TASKS_DEFAULTS["n_attempts"]=5
PRACTICE_TASKS_DEFAULTS["max_total_practice_runs"]=4000
PRACTICE_TASKS_DEFAULTS["n_random_actions"]=5
PRACTICE_TASKS_DEFAULTS["max_steps"]=50
PRACTICE_TASKS_DEFAULTS["lookback"]=8
PRACTICE_TASKS_DEFAULTS["executor"]="single_actions"
PRACTICE_TASKS_DEFAULTS["controller_variant"]="low_level"
PRACTICE_TASKS_DEFAULTS["checker_max_new_tokens"]=2000
PRACTICE_TASKS_DEFAULTS["max_concurrency"]=16

PRACTICE_TASKS_ARG_KEYS=("${PRACTICE_TASKS_ESSENTIALS[@]}" "${!PRACTICE_TASKS_DEFAULTS[@]}")

# safety_margin must match CREATE_DATASET's: the two slice each episode's call log at the same point.
CLEAN_PRACTICE_ESSENTIALS=("practice_path")
populate_array VLM_ESSENTIALS CLEAN_PRACTICE_ESSENTIALS
declare -A CLEAN_PRACTICE_DEFAULTS
populate_dict VLM_DEFAULTS CLEAN_PRACTICE_DEFAULTS
CLEAN_PRACTICE_DEFAULTS["k"]=3
CLEAN_PRACTICE_DEFAULTS["safety_margin"]=2
CLEAN_PRACTICE_DEFAULTS["max_concurrency"]=16

CLEAN_PRACTICE_ARG_KEYS=("${CLEAN_PRACTICE_ESSENTIALS[@]}" "${!CLEAN_PRACTICE_DEFAULTS[@]}")

CREATE_DATASET_ESSENTIALS=("practice_path")
declare -A CREATE_DATASET_DEFAULTS
CREATE_DATASET_DEFAULTS["overwrite"]=false
CREATE_DATASET_DEFAULTS["safety_margin"]=2
CREATE_DATASET_DEFAULTS["val_frac"]=0.2
CREATE_DATASET_DEFAULTS["seed"]=0

CREATE_DATASET_ARG_KEYS=("${CREATE_DATASET_ESSENTIALS[@]}" "${!CREATE_DATASET_DEFAULTS[@]}")

# "none" is a real value of balance (no downsampling), not an absent-argument sentinel.
MERGE_PRACTICES_ESSENTIALS=("practice_paths" "save_path")
declare -A MERGE_PRACTICES_DEFAULTS
MERGE_PRACTICES_DEFAULTS["balance"]="none"
MERGE_PRACTICES_DEFAULTS["seed"]=0

MERGE_PRACTICES_ARG_KEYS=("${MERGE_PRACTICES_ESSENTIALS[@]}" "${!MERGE_PRACTICES_DEFAULTS[@]}")

GUIDANCE_AND_PRACTICE_ESSENTIALS=("trajectory_path")
populate_array VLM_ESSENTIALS GUIDANCE_AND_PRACTICE_ESSENTIALS
declare -A GUIDANCE_AND_PRACTICE_DEFAULTS
populate_dict INFER_GUIDANCE_DEFAULTS GUIDANCE_AND_PRACTICE_DEFAULTS
populate_dict PRACTICE_TASKS_DEFAULTS GUIDANCE_AND_PRACTICE_DEFAULTS
populate_dict CLEAN_PRACTICE_DEFAULTS GUIDANCE_AND_PRACTICE_DEFAULTS
populate_dict CREATE_DATASET_DEFAULTS GUIDANCE_AND_PRACTICE_DEFAULTS
GUIDANCE_AND_PRACTICE_DEFAULTS["guidance_only"]=false
GUIDANCE_AND_PRACTICE_DEFAULTS["practice_only"]=false
GUIDANCE_AND_PRACTICE_DEFAULTS["do_clean"]=true

GUIDANCE_AND_PRACTICE_ARG_KEYS=("${GUIDANCE_AND_PRACTICE_ESSENTIALS[@]}" "${!GUIDANCE_AND_PRACTICE_DEFAULTS[@]}")

# source is required, not defaulted: it is the document's recorded provenance.
BUILD_INFO_ESSENTIALS=("trajectory_path" "source")
populate_array VLM_ESSENTIALS BUILD_INFO_ESSENTIALS
declare -A BUILD_INFO_DEFAULTS
populate_dict VLM_DEFAULTS BUILD_INFO_DEFAULTS
BUILD_INFO_DEFAULTS["n_frames"]=8
BUILD_INFO_DEFAULTS["max_concurrency"]=16
BUILD_INFO_DEFAULTS["stage"]="all"
# executor and controller_variant are provenance-only here; they name no path.
BUILD_INFO_DEFAULTS["executor"]="single_actions"
BUILD_INFO_DEFAULTS["controller_variant"]="low_level"
BUILD_INFO_DEFAULTS["overwrite_from_round"]=none
# Above the shared VLM default of 1000: truncation here silently drops the trailing insights.
BUILD_INFO_DEFAULTS["max_new_tokens"]=3000

BUILD_INFO_ARG_KEYS=("${BUILD_INFO_ESSENTIALS[@]}" "${!BUILD_INFO_DEFAULTS[@]}")

CREATE_INFO_DOC_ESSENTIALS=("source")
populate_array VLM_ESSENTIALS CREATE_INFO_DOC_ESSENTIALS
declare -A CREATE_INFO_DOC_DEFAULTS
populate_dict VLM_DEFAULTS CREATE_INFO_DOC_DEFAULTS
CREATE_INFO_DOC_DEFAULTS["run_name"]="my_run"
CREATE_INFO_DOC_DEFAULTS["executor"]="single_actions"
CREATE_INFO_DOC_DEFAULTS["controller_variant"]="low_level"
CREATE_INFO_DOC_DEFAULTS["n_frames"]=8
CREATE_INFO_DOC_DEFAULTS["max_concurrency"]=16
CREATE_INFO_DOC_DEFAULTS["max_new_tokens"]=3000
CREATE_INFO_DOC_DEFAULTS["overwrite_from_round"]=none

CREATE_INFO_DOC_ARG_KEYS=("${CREATE_INFO_DOC_ESSENTIALS[@]}" "${!CREATE_INFO_DOC_DEFAULTS[@]}")

CREATE_ALL_INFO_DOCS_ESSENTIALS=()
populate_array VLM_ESSENTIALS CREATE_ALL_INFO_DOCS_ESSENTIALS
declare -A CREATE_ALL_INFO_DOCS_DEFAULTS
populate_dict CREATE_INFO_DOC_DEFAULTS CREATE_ALL_INFO_DOCS_DEFAULTS
CREATE_ALL_INFO_DOCS_DEFAULTS["mode"]="both"

CREATE_ALL_INFO_DOCS_ARG_KEYS=("${CREATE_ALL_INFO_DOCS_ESSENTIALS[@]}" "${!CREATE_ALL_INFO_DOCS_DEFAULTS[@]}")

# The source verticals a --mode selects. Order matters: zeroshot reads first in a
# comma-joined --info_docs list.
function info_mode_sources() {
    case "$1" in
        curiosity_only) echo "curiosity" ;;
        zeroshot_only)  echo "zeroshot" ;;
        both)           echo "zeroshot curiosity" ;;
        *)              echo "" ;;
    esac
}


BUILD_INFO_ALL_ESSENTIALS=()
populate_array VLM_ESSENTIALS BUILD_INFO_ALL_ESSENTIALS
BUILD_INFO_ALL_ESSENTIALS+=("run_name")
declare -A BUILD_INFO_ALL_DEFAULTS
populate_dict VLM_DEFAULTS BUILD_INFO_ALL_DEFAULTS
BUILD_INFO_ALL_DEFAULTS["max_new_tokens"]=3000
BUILD_INFO_ALL_DEFAULTS["executor"]="single_actions"
# Paired with executor: the attempts dir the documents are built from is keyed on both.
BUILD_INFO_ALL_DEFAULTS["controller_variant"]="low_level"
BUILD_INFO_ALL_DEFAULTS["mode"]="both"
BUILD_INFO_ALL_DEFAULTS["stage"]="all"
BUILD_INFO_ALL_DEFAULTS["n_frames"]=8
BUILD_INFO_ALL_DEFAULTS["max_concurrency"]=16
# `none` sentinel: --overwrite_from_round is only emitted when set, since passing it rebuilds the merge tree.
BUILD_INFO_ALL_DEFAULTS["overwrite_from_round"]=none
BUILD_INFO_ALL_DEFAULTS["do_debug"]=true

BUILD_INFO_ALL_ARG_KEYS=("${BUILD_INFO_ALL_ESSENTIALS[@]}" "${!BUILD_INFO_ALL_DEFAULTS[@]}")

BENCHMARK_INFO_ALL_ESSENTIALS=()
populate_array VLM_ESSENTIALS BENCHMARK_INFO_ALL_ESSENTIALS
BENCHMARK_INFO_ALL_ESSENTIALS+=("run_name")
declare -A BENCHMARK_INFO_ALL_DEFAULTS=(
    # Both the executor the documents were built from and the one run at test time.
    ["executor"]="single_actions"
    ["controller_variant"]="low_level"
    ["mode"]="both"
    ["knowledge_mode"]="both"
    ["baseline"]=true
    ["max_steps"]=50
    ["max_concurrency"]=8
    ["supervisor_vlm_model"]=none
    ["supervisor_vlm_kind"]=none
    ["regenerate"]=false
)

BENCHMARK_INFO_ALL_ARG_KEYS=("${BENCHMARK_INFO_ALL_ESSENTIALS[@]}" "${!BENCHMARK_INFO_ALL_DEFAULTS[@]}")

INFO_FULL_ESSENTIALS=()
populate_array VLM_ESSENTIALS INFO_FULL_ESSENTIALS
INFO_FULL_ESSENTIALS+=("run_name")
declare -A INFO_FULL_DEFAULTS
populate_dict BUILD_INFO_ALL_DEFAULTS INFO_FULL_DEFAULTS
populate_dict BENCHMARK_INFO_ALL_DEFAULTS INFO_FULL_DEFAULTS
INFO_FULL_DEFAULTS["do_build"]=true
INFO_FULL_DEFAULTS["do_benchmark"]=true
INFO_FULL_DEFAULTS["max_steps"]=150
# The two stages want different concurrency, so the key is split; info_full.sh assigns the
# right half to max_concurrency before each forwarding call.
INFO_FULL_DEFAULTS["build_max_concurrency"]="${BUILD_INFO_ALL_DEFAULTS["max_concurrency"]}"
INFO_FULL_DEFAULTS["bench_max_concurrency"]="${BENCHMARK_INFO_ALL_DEFAULTS["max_concurrency"]}"
unset INFO_FULL_DEFAULTS["max_concurrency"]

INFO_FULL_ARG_KEYS=("${INFO_FULL_ESSENTIALS[@]}" "${!INFO_FULL_DEFAULTS[@]}")

# game is NOT essential here: a sweep discovers its own games.
INFO_FULL_ALL_GAMES_ESSENTIALS=("model_name" "vlm_kind" "run_name")   # VLM_ESSENTIALS - game
declare -A INFO_FULL_ALL_GAMES_DEFAULTS
populate_dict INFO_FULL_DEFAULTS INFO_FULL_ALL_GAMES_DEFAULTS
# mode is discovered per game from what is on disk, so it must not be user-settable here.
unset INFO_FULL_ALL_GAMES_DEFAULTS["mode"]
# Accepted and ignored: the loop overwrites it per game. Use --games to restrict the sweep.
INFO_FULL_ALL_GAMES_DEFAULTS["game"]=none
INFO_FULL_ALL_GAMES_DEFAULTS["games"]="auto"
INFO_FULL_ALL_GAMES_DEFAULTS["skip_games"]=none
INFO_FULL_ALL_GAMES_DEFAULTS["dry_run"]=false

INFO_FULL_ALL_GAMES_ARG_KEYS=("${INFO_FULL_ALL_GAMES_ESSENTIALS[@]}" "${!INFO_FULL_ALL_GAMES_DEFAULTS[@]}")

# run_name is required (no default) — removed from DEFAULTS after population
CURIOSITY_TASKS_ESSENTIALS=("run_name")
populate_array VLM_ESSENTIALS CURIOSITY_TASKS_ESSENTIALS
populate_array ENV_ESSENTIALS CURIOSITY_TASKS_ESSENTIALS

declare -A CURIOSITY_TASKS_DEFAULTS
populate_dict CREATE_TRAJ_DEFAULTS CURIOSITY_TASKS_DEFAULTS
populate_dict INFER_TASKS_DEFAULTS CURIOSITY_TASKS_DEFAULTS
unset CURIOSITY_TASKS_DEFAULTS["run_name"]

CURIOSITY_TASKS_ARG_KEYS=("${CURIOSITY_TASKS_ESSENTIALS[@]}" "${!CURIOSITY_TASKS_DEFAULTS[@]}")

PROPOSE_AND_ATTEMPT_ESSENTIALS=()
populate_array PROPOSE_ZEROSHOT_ESSENTIALS PROPOSE_AND_ATTEMPT_ESSENTIALS

declare -A PROPOSE_AND_ATTEMPT_DEFAULTS
populate_dict PROPOSE_ZEROSHOT_DEFAULTS PROPOSE_AND_ATTEMPT_DEFAULTS
populate_dict ATTEMPT_TASKS_DEFAULTS PROPOSE_AND_ATTEMPT_DEFAULTS
# Not inherited from PROPOSE_ZEROSHOT_DEFAULTS: proposal has no run_name, the curiosity lookup needs one.
PROPOSE_AND_ATTEMPT_DEFAULTS["run_name"]=all
PROPOSE_AND_ATTEMPT_DEFAULTS["propose_only"]=false
PROPOSE_AND_ATTEMPT_DEFAULTS["attempt_only"]=false

PROPOSE_AND_ATTEMPT_ARG_KEYS=("${PROPOSE_AND_ATTEMPT_ESSENTIALS[@]}" "${!PROPOSE_AND_ATTEMPT_DEFAULTS[@]}")

COLLECT_AND_INFO_ALL_ESSENTIALS=()
populate_array CREATE_ALL_INFO_DOCS_ESSENTIALS COLLECT_AND_INFO_ALL_ESSENTIALS

declare -A COLLECT_AND_INFO_ALL_DEFAULTS
populate_dict PROPOSE_AND_ATTEMPT_DEFAULTS COLLECT_AND_INFO_ALL_DEFAULTS
populate_dict CREATE_ALL_INFO_DOCS_DEFAULTS COLLECT_AND_INFO_ALL_DEFAULTS
COLLECT_AND_INFO_ALL_DEFAULTS["run_name"]="my_run"
# The two stages want different token budgets, so the info leg's is carried under its own key.
COLLECT_AND_INFO_ALL_DEFAULTS["max_new_tokens"]="${PROPOSE_AND_ATTEMPT_DEFAULTS["max_new_tokens"]}"
COLLECT_AND_INFO_ALL_DEFAULTS["info_max_new_tokens"]="${CREATE_ALL_INFO_DOCS_DEFAULTS["max_new_tokens"]}"

COLLECT_AND_INFO_ALL_ARG_KEYS=("${COLLECT_AND_INFO_ALL_ESSENTIALS[@]}" "${!COLLECT_AND_INFO_ALL_DEFAULTS[@]}")

# debug.py's group options, shared by every subcommand (scripts/debug/*.sh).
DEBUG_ESSENTIALS=()
populate_array ESSENTIAL_ARGS DEBUG_ESSENTIALS
declare -A DEBUG_DEFAULTS=(
    ["run_name"]="my_run"
    ["executor"]="single_actions"
    ["controller_variant"]="low_level"
    ["extra_name"]="none"
    ["output_dir"]="none"
    ["mode"]="both"
)
DEBUG_ARG_KEYS=("${DEBUG_ESSENTIALS[@]}" "${!DEBUG_DEFAULTS[@]}")

# Same, plus model_name, a *subcommand* option in debug.py that debug_group_flags does not emit.
DEBUG_MODEL_ESSENTIALS=("model_name")
populate_array DEBUG_ESSENTIALS DEBUG_MODEL_ESSENTIALS
declare -A DEBUG_MODEL_DEFAULTS
populate_dict DEBUG_DEFAULTS DEBUG_MODEL_DEFAULTS

DEBUG_MODEL_ARG_KEYS=("${DEBUG_MODEL_ESSENTIALS[@]}" "${!DEBUG_MODEL_DEFAULTS[@]}")

# Build the flag string for debug.py's group options from an ARGS dict. --output_dir and
# --extra_name are omitted when `none`; --overwrite is always passed.
function debug_group_flags() {
    local -n _dict="$1"
    local result="--game ${_dict["game"]} --run_name ${_dict["run_name"]} --executor ${_dict["executor"]} --controller_variant ${_dict["controller_variant"]} --mode ${_dict["mode"]} --overwrite"
    if [[ "${_dict["output_dir"]}" != "none" ]]; then
        result+=" --output_dir ${_dict["output_dir"]}"
    fi
    if [[ "${_dict["extra_name"]}" != "none" ]]; then
        result+=" --extra_name ${_dict["extra_name"]}"
    fi
    echo "$result"
}

RUN_BENCHMARK_ESSENTIALS=()
populate_array ESSENTIAL_ARGS RUN_BENCHMARK_ESSENTIALS

declare -A RUN_BENCHMARK_DEFAULTS=(
    ["supervisor"]="dummy"
    ["controller_variant"]="low_level"
    # Free-text discriminator appended to the run name. `none` means absent.
    ["extra_name"]="none"
    ["executor"]="single_none"
    ["executor_vlm_model"]="Qwen/Qwen3-VL-8B-Instruct"
    ["executor_vlm_kind"]="huggingface"
    ["supervisor_vlm_model"]=none
    ["supervisor_vlm_kind"]=none
    ["supervisor_max_new_tokens"]=5000
    ["save_video"]=true
    ["max_steps"]=175
    ["n_tasks"]=none
    ["verbose"]=false
    ["regenerate"]=false
    ["max_leg_steps"]=5
    ["max_attempts_per_step"]=3
    ["max_replans"]=2
    ["max_frames_per_slice"]=8
    ["info_docs"]=none
    ["parametric_categories"]=10
    ["max_concurrency"]=8
    ["executor_max_new_tokens"]=8000
    # RL run_name whose world model drives the `world_model` executor. `none` means absent.
    ["world_model_run_name"]=none
    # Game whose checkpoint the `world_model` executor borrows. `none` means the benchmark game's own.
    ["world_model_game"]=none
)

RUN_BENCHMARK_ARG_KEYS=("${RUN_BENCHMARK_ESSENTIALS[@]}" "${!RUN_BENCHMARK_DEFAULTS[@]}")

RUN_BENCHMARK_INFO_RETRIEVAL_ESSENTIALS=()
populate_array RUN_BENCHMARK_ESSENTIALS RUN_BENCHMARK_INFO_RETRIEVAL_ESSENTIALS

declare -A RUN_BENCHMARK_INFO_RETRIEVAL_DEFAULTS
populate_dict RUN_BENCHMARK_DEFAULTS RUN_BENCHMARK_INFO_RETRIEVAL_DEFAULTS
# Both are derived, not settable.
unset RUN_BENCHMARK_INFO_RETRIEVAL_DEFAULTS["supervisor"]
unset RUN_BENCHMARK_INFO_RETRIEVAL_DEFAULTS["info_docs"]
# Which documents to read, NOT which model is benchmarked (that is executor_vlm_model).
# docs_run_name selects the curiosity leg's input; docs_executor and docs_controller_variant
# select the zeroshot leg's.
RUN_BENCHMARK_INFO_RETRIEVAL_DEFAULTS["docs_model"]="google/gemma-4-31b-it"
RUN_BENCHMARK_INFO_RETRIEVAL_DEFAULTS["docs_mode"]="both"
RUN_BENCHMARK_INFO_RETRIEVAL_DEFAULTS["docs_run_name"]="my_run"
RUN_BENCHMARK_INFO_RETRIEVAL_DEFAULTS["docs_executor"]="single_actions"
RUN_BENCHMARK_INFO_RETRIEVAL_DEFAULTS["docs_controller_variant"]="low_level"

RUN_BENCHMARK_INFO_RETRIEVAL_ARG_KEYS=("${RUN_BENCHMARK_INFO_RETRIEVAL_ESSENTIALS[@]}" "${!RUN_BENCHMARK_INFO_RETRIEVAL_DEFAULTS[@]}")
