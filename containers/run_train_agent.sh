#!/usr/bin/env bash
# Run prime-agent in a train sandbox, developing a harness on its own, in the fixed cycle that
# containers/train_agent/cycle.py drives, until both games are cleared, /work/STOP exists, the spending cap is
# reached, or --time_limit passes.
#
#     bash containers/run_train_agent.sh --name <run> --harness strategist --model google/gemini-3.6-flash \
#         --max_cost 1000 [--from_run <earlier run>] [--provider openrouter|google]
#
# A run is named once and kept in <runs_root>/<run> (default: $storage_dir/env_server/runs). A new name starts
# from the current main of harnesses/<harness> on a branch of its own, agent/<run>, or with --from_run from the
# harness branch of that earlier run; an existing name continues where it stopped, and must name the harness
# it started from (recorded in <run_dir>/run.json, with the commit it started from and every launch). The
# starting harness itself is never changed by a run.
#
# The run's branches are pushed to runs/<run>/ at harness_remote (configs/project_vars.yaml), or --push_url;
# --no_push keeps them in the run directory only. The agent's rules are containers/train_agent/AGENTS.md and
# its step prompts containers/train_agent/prompts/. To stop it: touch <run_dir>/work/STOP. Exits 0 when the
# server confirms both games cleared, 1 when the agent's calls keep failing at once.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source configs/config.env
source containers/common.sh

NAME=""
HARNESS=strategist
RUNS_ROOT="$storage_dir/env_server/runs"
MODEL=""
MAX_COST=1000
TIME_LIMIT=""
PUSH_URL="${harness_remote:-}"
FROM_RUN=""
PROVIDER=openrouter
PYTHON=setup/.venv/bin/python

usage() { sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) NAME="$2"; shift 2 ;;
        --harness) HARNESS="$2"; shift 2 ;;
        --runs_root) RUNS_ROOT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --max_cost) MAX_COST="$2"; shift 2 ;;
        --time_limit) TIME_LIMIT="$2"; shift 2 ;;
        --push_url) PUSH_URL="$2"; shift 2 ;;
        --no_push) PUSH_URL=""; shift ;;
        --from_run) FROM_RUN="$2"; shift 2 ;;
        --provider) PROVIDER="$2"; shift 2 ;;
        *) usage ;;
    esac
done
[[ -n "$NAME" ]] || die "--name is required"
[[ "$NAME" =~ ^[A-Za-z0-9_]+$ ]] || die "--name is letters, digits and underscores"
[[ -n "$MODEL" ]] || die "--model is required"
HARNESS_REPO="harnesses/$HARNESS"
[[ -d "$HARNESS_REPO/.git" ]] \
    || die "no harness $HARNESS at $HARNESS_REPO; make it with containers/harness_repo.sh sync $HARNESS"
RUN_DIR="$RUNS_ROOT/$NAME"
mkdir -p "$RUN_DIR"
if [[ -n "$FROM_RUN" && ! -d "$RUN_DIR/work/harness/.git" ]]; then
    SOURCE="$RUNS_ROOT/$FROM_RUN/work/harness"
    [[ -d "$SOURCE/.git" ]] || die "no harness of an earlier run at $SOURCE"
    [[ -z "$(git -C "$SOURCE" status --porcelain)" ]] || die "$SOURCE has uncommitted changes; commit them first"
    mkdir -p "$RUN_DIR/work"
    git clone -q --no-tags "$SOURCE" "$RUN_DIR/work/harness"
    git -C "$RUN_DIR/work/harness" fetch -q --no-tags origin upstream:upstream
    git -C "$RUN_DIR/work/harness" remote remove origin
    git -C "$RUN_DIR/work/harness" branch -q -m "agent/$NAME"
    git -C "$RUN_DIR/work/harness" config user.name "train agent"
    git -C "$RUN_DIR/work/harness" config user.email "agent@train-sandbox"
    echo "run $NAME: harness copied from run $FROM_RUN at $(git -C "$RUN_DIR/work/harness" rev-parse --short HEAD)"
fi

"$PYTHON" - "$RUN_DIR" "$NAME" "$HARNESS" "$HARNESS_REPO" "$MODEL" "$MAX_COST" "$PUSH_URL" <<'PY'
import datetime, json, os, subprocess, sys
run_dir, name, harness, repo, model, max_cost, push_url = sys.argv[1:]
path = os.path.join(run_dir, "run.json")
worktree = os.path.join(run_dir, "work", "harness")
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def git(*args, cwd):
    return subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True, check=True).stdout.strip()

if os.path.exists(path):
    record = json.load(open(path))
    if record["harness"] != harness:
        sys.exit(f"error: run {name} started from harness {record['harness']!r}, not {harness!r}")
else:
    if os.path.isdir(os.path.join(worktree, ".git")):
        base = git("merge-base", "HEAD", "upstream", cwd=worktree)
    else:
        base = git("rev-parse", "main", cwd=repo)
    record = {"name": name, "harness": harness, "base_commit": base, "branch": f"agent/{name}",
              "created": now, "launches": []}
record["launches"].append({"t": now, "model": model, "max_cost": float(max_cost), "push_url": push_url or None})
with open(path + ".tmp", "w") as f:
    json.dump(record, f, indent=1)
os.replace(path + ".tmp", path)
print(f"run {name}: harness {harness} from {record['base_commit'][:7]}, launch {len(record['launches'])}")
PY

args=(--kind train --run_dir "$RUN_DIR" --model "$MODEL" --max_cost "$MAX_COST" --harness_repo "$HARNESS_REPO"
      --provider "$PROVIDER")
[[ -z "$PUSH_URL" ]] || args+=(--push_url "$PUSH_URL" --push_prefix "runs/$NAME")
[[ -z "$TIME_LIMIT" ]] || args+=(--time_limit "$TIME_LIMIT")
exec bash containers/run_sandbox.sh "${args[@]}" -- /opt/harness/bin/python /opt/train/cycle.py
