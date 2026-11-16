#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 OUTPUT" >&2
    exit 2
fi
root=$1
python=${KXVO_PYTHON:-.venv/bin/python}
binary=${KXVO_ARDUCOPTER:-runs/ardupilot/build/sitl/bin/arducopter}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
mkdir -p "$root"
run() {
    local label=$1 rc=0
    [[ ! -f "$root/$label/summary.json" ]] || return 0
    "$python" -m kxvo.sitl --arducopter "$binary" --output "$root/$label" \
        --instance 9 --clock autopilot --seed 1 --delay-ms 80 --jitter-ms 15 --drop 0.1 \
        --exposure-ms 12 --blur-min-px 7 --blackout-seconds 1 --downlink-bps 1250 \
        --fallback-source flow > "$root/$label.log" 2>&1 || rc=$?
    [[ -f "$root/$label/summary.json" ]] || return "$rc"
    echo "$label exit $rc"
}
for trial in 1 2 3; do run "idle-$trial"; done
yes >/dev/null &
worker1=$!
yes >/dev/null &
worker2=$!
trap 'kill "$worker1" "$worker2" 2>/dev/null || true' EXIT
for trial in 1 2 3; do run "loaded-$trial"; done
