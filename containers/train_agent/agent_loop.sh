#!/bin/sh
set -u
PASS_SECONDS="${TRAIN_PASS_SECONDS:-10800}"
FRESH_SECONDS="${TRAIN_FRESH_SECONDS:-43200}"
IDLE_SECONDS="${TRAIN_IDLE_SECONDS:-600}"
WATCH_GAP_SECONDS="${TRAIN_WATCH_GAP_SECONDS:-240}"
QUICK_SECONDS=120
QUICK_BEFORE_FRESH=3
QUICK_BEFORE_STOP=6
PY=/opt/harness/bin/python
SESSIONS="$HOME/.prime/agent/sessions"

mkdir -p /work/logs /work/backups /work/agent_logs
log=/work/agent_logs/loop.log
note=/work/agent_logs/next_prompt_note.md
count_file=/work/agent_logs/passes
conversation_file=/work/agent_logs/conversation_started
n=0
[ -f "$count_file" ] && n="$(cat "$count_file")"
quick=0
fresh_next=0

playthrough_running() {
    ps -eo args 2>/dev/null | grep -q '[r]un_playthrough.py'
}

session_age() {
    latest="$(ls -t "$SESSIONS"/*.jsonl 2>/dev/null | head -1)"
    if [ -z "$latest" ]; then
        echo 0
        return
    fi
    echo $(( $(date +%s) - $(stat -c %Y "$latest") ))
}

while [ ! -e /work/STOP ]; do
    now="$(date +%s)"
    conversation_started=0
    [ -f "$conversation_file" ] && conversation_started="$(cat "$conversation_file")"
    if [ "$fresh_next" -eq 1 ] || [ "$n" -eq 0 ] || [ $((now - conversation_started)) -ge "$FRESH_SECONDS" ]; then
        prompt="$(cat /opt/train/kickoff.md)"
        echo "$now" > "$conversation_file"
        set --
    else
        prompt="$(cat /opt/train/continue.md)"
        set -- --continue
    fi
    fresh_next=0
    if [ -s "$note" ]; then
        prompt="$prompt

$(cat "$note")"
        rm -f "$note"
    fi
    started="$(date +%s)"
    echo "pass $n started $(date -u +%FT%TZ)" >> "$log"
    timeout --signal=TERM --kill-after=60 "$PASS_SECONDS" \
        prime-agent --offline --provider gateway --model "$SANDBOX_MODEL" --cwd /work "$@" -p "$prompt" \
        > "/work/agent_logs/pass_$n.log" 2>&1 &
    pid=$!
    idle=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep 30
        age="$(session_age)"
        if [ "$age" -ge "$IDLE_SECONDS" ] && playthrough_running && kill -0 "$pid" 2>/dev/null; then
            echo "pass $n idle for ${age}s with a playthrough running; ending it" >> "$log"
            idle="$age"
            kill -TERM "$pid" 2>/dev/null
            break
        fi
    done
    wait "$pid"
    status=$?
    elapsed=$(( $(date +%s) - started ))
    echo "pass $n ended with status $status after ${elapsed}s" >> "$log"
    n=$((n + 1))
    echo "$n" > "$count_file"

    if [ -e /work/GIVE_UP ]; then
        mv /work/GIVE_UP /work/agent_logs/gave_up.md
        echo "stopping: the agent gave up; its reasons are in agent_logs/gave_up.md" >> "$log"
        exit 3
    fi
    if [ -e /work/FINISHED ]; then
        if verdict="$("$PY" /opt/train/check_finish.py $TRAIN_GAMES 2>&1)"; then
            mv /work/FINISHED /work/agent_logs/finished.md
            echo "stopping: finished, and the server confirms it: $verdict" | tr '\n' ' ' >> "$log"
            echo >> "$log"
            exit 0
        fi
        mv /work/FINISHED "/work/agent_logs/rejected_finish_$n.md"
        echo "finish claim rejected after pass $((n - 1)): $verdict" | tr '\n' ' ' >> "$log"
        echo >> "$log"
        cat > "$note" <<EOF
Your claim that the work is finished (/work/FINISHED, now kept as /work/agent_logs/rejected_finish_$n.md) was rejected. The server's record of cleared games says:
$verdict
A game counts as cleared only when the server has seen a playthrough beat its Champion. Keep working on the games that are not cleared.
EOF
    fi

    if [ "$idle" -gt 0 ] || playthrough_running; then
        quick=0
        cat >> "$note" <<EOF
Your last session ended its turn while a playthrough was still running$( [ "$idle" -gt 0 ] && echo ", and sat idle for ${idle}s, so it was ended for you" ). /work/AGENTS.md says not to end your turn while a playthrough runs. Check whether it is still running (a playthrough started without setsid stops when a session is ended; restart it with the same --name), record a check with /opt/train/watch_check.py, and keep watching it every 5 minutes.
EOF
        sleep "$WATCH_GAP_SECONDS"
        continue
    fi
    if [ "$elapsed" -lt "$QUICK_SECONDS" ]; then
        quick=$((quick + 1))
        if [ "$quick" -ge "$QUICK_BEFORE_STOP" ]; then
            echo "stopping: $quick passes in a row ended within ${QUICK_SECONDS}s" >> "$log"
            exit 1
        fi
        if [ "$quick" -ge "$QUICK_BEFORE_FRESH" ]; then
            fresh_next=1
            cat >> "$note" <<EOF
Your last $quick sessions each ended within ${QUICK_SECONDS} seconds without starting any new work. The games are not cleared, so the work is not done. Run /opt/harness/bin/python /opt/train/progress.py to see how far every playthrough has got, pick the point where the furthest one stalls, and work on getting past it. If you truly believe you have run out of ideas, give up the way /work/AGENTS.md describes rather than ending sessions early.
EOF
        fi
        sleep 60
    else
        quick=0
    fi
done
echo "stopping: /work/STOP exists" >> "$log"
