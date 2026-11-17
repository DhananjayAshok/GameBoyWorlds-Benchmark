#!/bin/bash
JOBID="$1"; CPUS="$2"; LOG="$3"; shift 3
for v in $(compgen -e | grep '^SLURM_'); do unset "$v"; done
mkdir -p "$(dirname "$LOG")"
setsid nohup srun --jobid="$JOBID" --overlap -N1 -n1 -c "$CPUS" --mem=0 "$@" > "$LOG" 2>&1 < /dev/null &
echo "launched step in job $JOBID (pid $!) -> $LOG"
