#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 OUTPUT [cpu|mps|cuda]" >&2
    exit 2
fi
root=$1
device=${2:-cpu}
py=${KXVO_PYTHON:-.venv/bin/python}
dataset="$root/dataset"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
scripts/setup-xfeat.sh
if [[ ! -s "$dataset/pairs.csv" ]]; then
    "$py" -m kxvo.tartanair --plan experiments/tartanair-geometry.json --output "$dataset" --neighbor-offset 2
fi
for arm in geometric warp; do
    args=()
    if [[ "$arm" == warp ]]; then
        args=(--warp-supervision)
    fi
    "$py" -m kxvo.train --manifest "$dataset/manifest.csv" --geometric-pairs "$dataset/pairs.csv" \
        --output "$root/$arm" --checkpoint runs/xfeat-upstream/weights/xfeat.pt --device "$device" \
        --train-detector --width 320 --height 320 --seeds 1 2 --steps 1000 --validate-every 500 \
        "${args[@]+"${args[@]}"}"
done
