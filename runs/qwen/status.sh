#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
echo "=== hold jobs / steps"
squeue -u "$USER" -s -o "%.18i %.12P %.30j %.10M %N" | grep -v "^ *[0-9]*\.\(batch\|extern\)"
echo
echo "=== lanes (last event)"
for f in logs/qwen/lane_*.out; do
    printf "%-55s %s\n" "$(basename "$f" .out)" "$(grep '^\[' "$f" | tail -1)"
done
echo
echo "=== progress bars (latest per stage log)"
for d in logs/qwen/lanes/*/; do
    for f in "$d"*.out; do
        [ -f "$f" ] || continue
        bar=$(tr '\r' '\n' < "$f" | grep -E '[0-9]+/[0-9]+ \[' | tail -1 | sed -E 's/^.*\|[^|]*\| *//' | cut -c1-60)
        printf "%-45s %-28s %s\n" "$(basename "$d")" "$(basename "$f" .out)" "$bar"
    done
done
echo
echo "=== new Qwen result CSVs"
ls -la --time-style=+%m-%d_%H:%M results/benchmark/*/*qwen3-vl-*curiosity_only*.csv results/benchmark/*/*practice_*qwen3-vl*.csv 2>/dev/null || echo "  (none yet)"
