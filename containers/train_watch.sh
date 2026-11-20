#!/usr/bin/env bash
# Keeps a train run's work safe from the agent, on the host. Started by run_sandbox.sh --kind train.
#
#   train_watch.sh once|loop <run_dir> [<push_url> <push_prefix>]
#
# Each round fetches every branch of <run_dir>/work/harness into <run_dir>/harness_mirror.git (a history the
# agent cannot rewrite there: a rewritten branch is kept beside the old one under rewritten/<time>/) and, with
# <push_url>, pushes the mirror's branches but upstream (a copy of the starting harness, which is published as
# harness/<name>) to <push_prefix>/<branch> there. Once an hour, and on `once`, it
# also snapshots the agent's artifacts and notes into <run_dir>/archive/<time>/, hard-linking unchanged files.

set -uo pipefail
export GIT_TERMINAL_PROMPT=0

MODE="$1"
RUN="$2"
PUSH_URL="${3:-}"
PUSH_PREFIX="${4:-}"
WORK="$RUN/work"
MIRROR="$RUN/harness_mirror.git"
ARCHIVE="$RUN/archive"
LOG="$RUN/watch.log"
ROUND_SECONDS="${TRAIN_WATCH_SECONDS:-600}"
ARCHIVE_SECONDS="${TRAIN_ARCHIVE_SECONDS:-3600}"
ARCHIVED=(storage logs agent_logs playthroughs tests iterations cycle)

say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

mirror() {
    [[ -d "$WORK/harness/.git" ]] || return 0
    [[ -d "$MIRROR" ]] || git init -q --bare "$MIRROR"
    if ! git -C "$MIRROR" fetch -q --no-tags "$WORK/harness" 'refs/heads/*:refs/heads/*' 2>>"$LOG"; then
        local stamp
        stamp="$(date -u +%Y%m%dT%H%M%SZ)"
        git -C "$MIRROR" fetch -q --no-tags "$WORK/harness" "+refs/heads/*:refs/heads/rewritten/$stamp/*" 2>>"$LOG" \
            && say "a branch was rewritten; its new history is kept under rewritten/$stamp/"
    fi
    if [[ -n "$PUSH_URL" ]]; then
        local refspecs=()
        while read -r branch; do
            [[ "$branch" == upstream ]] || refspecs+=("refs/heads/$branch:refs/heads/$PUSH_PREFIX/$branch")
        done < <(git -C "$MIRROR" for-each-ref --format='%(refname:lstrip=2)' refs/heads)
        [[ ${#refspecs[@]} -gt 0 ]] || return 0
        git -C "$MIRROR" push -q "$PUSH_URL" "${refspecs[@]}" 2>>"$LOG" || say "push to $PUSH_URL failed"
    fi
}

archive() {
    local stamp last sources=()
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$ARCHIVE"
    last="$(ls -1 "$ARCHIVE" 2>/dev/null | tail -n 1)"
    for item in "${ARCHIVED[@]}"; do
        [[ -e "$WORK/$item" ]] && sources+=("$WORK/$item")
    done
    [[ ${#sources[@]} -gt 0 ]] || return 0
    local link=()
    [[ -n "$last" ]] && link=(--link-dest "$ARCHIVE/$last")
    rsync -a --exclude __pycache__ "${link[@]}" "${sources[@]}" "$ARCHIVE/$stamp/" 2>>"$LOG" \
        && say "archived to archive/$stamp"
}

case "$MODE" in
    once)
        mirror
        archive ;;
    loop)
        last_archive=0
        while true; do
            mirror
            now="$(date +%s)"
            if (( now - last_archive >= ARCHIVE_SECONDS )); then
                archive
                last_archive="$now"
            fi
            sleep "$ROUND_SECONDS"
        done ;;
    *)
        sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 1 ;;
esac
