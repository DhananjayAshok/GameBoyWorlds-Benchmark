#!/usr/bin/env bash
# Harness repositories for the train sandbox: each is its own git repository with a fresh history, so an
# agent that sees all of it sees nothing of this repo's history.
#
#   sync <name> [--ref <git ref>]   build harnesses/<name> from <ref> of this repo (default train_only), with
#                                   containers/train_agent/harness_docs/<name>.md as its HARNESS.md, and
#                                   commit it on the harness's main branch; run again after changing <ref>.
#                                   HARNESS_DOC_REF=<ref> takes HARNESS.md from another ref
#   publish <name> [--url <git url>]
#                                   push harnesses/<name>'s main to harness/<name> at <url> (default:
#                                   harness_remote in configs/project_vars.yaml)
#   prepare <repo> <worktree> <branch> <note_file>
#                                   used by run_sandbox.sh: clone <repo> into <worktree> on <branch> the first
#                                   time; afterwards commit anything uncommitted and merge <repo>'s main
#                                   into it, writing <note_file> for the agent when that cannot be done cleanly
#
# A harness is used as `cd /work/harness; python run_playthrough.py --remote --game <game> --name <name>
# --model <model> --vlm_kind gateway --record_dir <dir> [--max_episodes <n>] [--start_checkpoint <run>:<id>]`.
# In <dir> it keeps checkpoints.jsonl (id, t, episode, steps, server_save, memory: every point the run can be
# forked from with --start_checkpoint) and trajectory.jsonl (t, episode, layer, kind, steps, text: its
# intentions, plans, attempts and outcomes). containers/train_agent/cycle.py relies on both.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source configs/config.env
source containers/common.sh

HARNESS_PATHS=(execution utils python_scripts run_playthrough.py agent_workspace/envclient.py README.md README_dev.md)
HELD_OUT_PATTERN='brown|prism'
AGENT_NAME="train agent"
AGENT_EMAIL="agent@train-sandbox"
PYTHON=setup/.venv/bin/python

git_agent() { git -C "$1" -c user.name="$AGENT_NAME" -c user.email="$AGENT_EMAIL" "${@:2}"; }

sync_harness() {
    local name="$1" ref="$2"
    local repo="harnesses/$name"
    local sha
    sha="$(git rev-parse --verify "$ref^{commit}")" || die "no such ref $ref"
    local stage
    stage="$(mktemp -d "$(sandbox_runtime_dir)/harness_stage.XXXXXX")"
    git archive "$sha" "${HARNESS_PATHS[@]}" | tar -x -C "$stage"
    local doc_ref="${HARNESS_DOC_REF:-$sha}"
    git show "$doc_ref:containers/train_agent/harness_docs/$name.md" > "$stage/HARNESS.md" 2>/dev/null \
        || { rm -rf "$stage"; die "$doc_ref has no containers/train_agent/harness_docs/$name.md"; }
    mkdir -p "$stage/configs"
    "$PYTHON" - "$stage/configs" "$sha" <<'PY'
import subprocess, sys, yaml
out, sha = sys.argv[1], sys.argv[2]
project = yaml.safe_load(subprocess.run(["git", "show", f"{sha}:configs/project_vars.yaml"],
                                        capture_output=True, text=True, check=True).stdout)
project["results_dir"] = "/work/results"
yaml.safe_dump(project, open(f"{out}/project_vars.yaml", "w"))
yaml.safe_dump({"storage_dir": "/work/storage"}, open(f"{out}/private_vars.yaml", "w"))
PY
    printf '__pycache__/\n*.py[cod]\n' > "$stage/.gitignore"
    if grep -rIl -i -E "$HELD_OUT_PATTERN" "$stage" >/dev/null; then
        grep -rIl -i -E "$HELD_OUT_PATTERN" "$stage" | sed "s|$stage/||" >&2
        rm -rf "$stage"
        die "$ref mentions a held-out game (files above); a harness repo must not"
    fi
    if [[ ! -d "$repo/.git" ]]; then
        mkdir -p "$repo"
        git -C "$repo" init -q -b main
    fi
    rsync -a --delete --exclude .git "$stage/" "$repo/"
    rm -rf "$stage"
    git -C "$repo" add -A
    if git -C "$repo" diff --cached --quiet; then
        echo "harnesses/$name already matches $ref ($sha)"
        return
    fi
    git -C "$repo" commit -q -m "Harness from GameBoyRL $ref"
    echo "harnesses/$name main is now $(git -C "$repo" rev-parse --short HEAD), from $ref ($sha)"
}

publish_harness() {
    local name="$1" url="$2"
    [[ -d "harnesses/$name/.git" ]] || die "no harness repository at harnesses/$name"
    [[ -n "$url" ]] || die "no --url and no harness_remote in configs/project_vars.yaml"
    GIT_TERMINAL_PROMPT=0 git -C "harnesses/$name" push -q "$url" "main:refs/heads/harness/$name"
    echo "published harnesses/$name main ($(git -C "harnesses/$name" rev-parse --short main)) as harness/$name"
}

prepare_worktree() {
    local repo="$1" worktree="$2" branch="$3" note="$4"
    repo="$(cd "$repo" && pwd)"
    if [[ ! -d "$worktree/.git" ]]; then
        [[ ! -e "$worktree" ]] || die "$worktree exists but is not a git repository"
        git clone -q --no-tags "$repo" "$worktree"
        git -C "$worktree" remote remove origin
        git -C "$worktree" branch -q -m main upstream
        git -C "$worktree" checkout -q -b "$branch"
        git -C "$worktree" config user.name "$AGENT_NAME"
        git -C "$worktree" config user.email "$AGENT_EMAIL"
        echo "harness: new $branch from $(git -C "$worktree" rev-parse --short HEAD)"
        return
    fi
    if [[ -n "$(git -C "$worktree" status --porcelain)" ]]; then
        if [[ -e "$worktree/.git/MERGE_HEAD" ]]; then
            git -C "$worktree" merge --abort || true
        fi
        git_agent "$worktree" add -A
        git_agent "$worktree" commit -q -m "Uncommitted work at the end of a launch, committed by the launcher" \
            || true
    fi
    git -C "$worktree" fetch -q --no-tags "$repo" "+main:upstream" 2>/dev/null \
        || git -C "$worktree" fetch -q --no-tags --update-head-ok "$repo" "+main:upstream"
    if git -C "$worktree" merge-base --is-ancestor upstream HEAD; then
        echo "harness: $(git -C "$worktree" rev-parse --abbrev-ref HEAD) already has upstream"
        return
    fi
    if git_agent "$worktree" merge -q --no-edit -m "Merge upstream harness changes" upstream >/dev/null 2>&1; then
        echo "harness: merged upstream $(git -C "$worktree" rev-parse --short upstream) into $(git -C "$worktree" rev-parse --abbrev-ref HEAD)"
        cat >> "$note" <<EOF
Changes from outside the sandbox were merged into your harness (branch upstream, commit $(git -C "$worktree" rev-parse --short upstream)). Read them with \`git log -p HEAD^2 --not HEAD^1\` in /work/harness before you go on.
EOF
        return
    fi
    git -C "$worktree" merge --abort || true
    echo "harness: upstream does not merge cleanly into $(git -C "$worktree" rev-parse --abbrev-ref HEAD); left for the agent"
    cat >> "$note" <<EOF
There are changes from outside the sandbox on branch upstream (commit $(git -C "$worktree" rev-parse --short upstream)) that do not merge cleanly into your branch. Merge them yourself first (\`git merge upstream\` in /work/harness), resolving each conflict so that both your fix and theirs survive, and commit the merge.
EOF
}

case "${1:-}" in
    sync)
        [[ $# -ge 2 ]] || die "usage: $0 sync <name> [--ref <git ref>]"
        name="$2"; ref=train_only
        [[ "${3:-}" == --ref ]] && ref="${4:?--ref needs a value}"
        sync_harness "$name" "$ref" ;;
    publish)
        [[ $# -ge 2 ]] || die "usage: $0 publish <name> [--url <git url>]"
        url="${harness_remote:-}"
        [[ "${3:-}" == --url ]] && url="${4:?--url needs a value}"
        publish_harness "$2" "$url" ;;
    prepare)
        [[ $# -eq 5 ]] || die "usage: $0 prepare <repo> <worktree> <branch> <note_file>"
        prepare_worktree "$2" "$3" "$4" "$5" ;;
    *)
        sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 1 ;;
esac
