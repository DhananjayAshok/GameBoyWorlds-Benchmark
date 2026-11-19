#!/bin/sh
set -u
PASS_SECONDS="${TRAIN_PASS_SECONDS:-10800}"
FRESH_EVERY="${TRAIN_FRESH_EVERY:-6}"
MAX_QUICK_FAILURES=5
QUICK_SECONDS=120

mkdir -p /work/logs /work/backups /work/agent_logs
count_file=/work/agent_logs/passes
n=0
[ -f "$count_file" ] && n="$(cat "$count_file")"
quick=0

while [ ! -e /work/STOP ]; do
    if [ $((n % FRESH_EVERY)) -eq 0 ]; then
        prompt="$(cat /opt/train/kickoff.md)"
        set --
    else
        prompt="$(cat /opt/train/continue.md)"
        set -- --continue
    fi
    started="$(date +%s)"
    echo "pass $n started $(date -u +%FT%TZ)" >> /work/agent_logs/loop.log
    timeout --signal=TERM --kill-after=60 "$PASS_SECONDS" \
        prime-agent --offline --provider gateway --model "$SANDBOX_MODEL" --cwd /work "$@" -p "$prompt" \
        > "/work/agent_logs/pass_$n.log" 2>&1
    status=$?
    elapsed=$(( $(date +%s) - started ))
    echo "pass $n ended with status $status after ${elapsed}s" >> /work/agent_logs/loop.log
    n=$((n + 1))
    echo "$n" > "$count_file"
    if [ "$elapsed" -lt "$QUICK_SECONDS" ]; then
        quick=$((quick + 1))
        if [ "$quick" -ge "$MAX_QUICK_FAILURES" ]; then
            echo "stopping: $quick passes in a row ended within ${QUICK_SECONDS}s" >> /work/agent_logs/loop.log
            exit 1
        fi
        sleep 60
    else
        quick=0
    fi
done
echo "stopping: /work/STOP exists" >> /work/agent_logs/loop.log
