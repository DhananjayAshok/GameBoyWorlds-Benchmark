#!/bin/sh
set -u
PASS_SECONDS="${TRAIN_PASS_SECONDS:-10800}"
FRESH_EVERY="${TRAIN_FRESH_EVERY:-6}"
QUICK_SECONDS=120
QUICK_BEFORE_FRESH=3
QUICK_BEFORE_STOP=6
PY=/opt/harness/bin/python

mkdir -p /work/logs /work/backups /work/agent_logs
log=/work/agent_logs/loop.log
note=/work/agent_logs/next_prompt_note.md
count_file=/work/agent_logs/passes
n=0
[ -f "$count_file" ] && n="$(cat "$count_file")"
quick=0
fresh_next=0

while [ ! -e /work/STOP ]; do
    if [ "$fresh_next" -eq 1 ] || [ $((n % FRESH_EVERY)) -eq 0 ]; then
        prompt="$(cat /opt/train/kickoff.md)"
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
        > "/work/agent_logs/pass_$n.log" 2>&1
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
