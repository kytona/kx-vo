#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 OUTPUT [cpu|mps|cuda] [STEPS]" >&2
    exit 2
fi
output=$1
device=${2:-cpu}
steps=${3:-500}
python=${KXVO_PYTHON:-.venv/bin/python}
dataset=runs/datasets/tartanair-xfeat
checkpoint=runs/xfeat-upstream/weights/xfeat.pt
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

scripts/setup-xfeat.sh
if [[ ! -f "$dataset/manifest.csv" ]]; then
    "$python" -m kxvo.tartanair --plan experiments/tartanair-xfeat.json --output "$dataset"
fi
"$python" -m kxvo.train --manifest "$dataset/manifest.csv" --output "$output" \
    --checkpoint "$checkpoint" --device "$device" --width 320 --height 320 \
    --train-detector --seeds 1 2 --steps "$steps" --validate-every 250

candidates=("$checkpoint")
for seed in 1 2; do
    folder="$output/seed-$seed"
    if [[ -f "$folder/selected.json" ]]; then
        name=$("$python" -c 'import json, sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$folder/selected.json")
        candidates+=("$folder/$name")
    else
        echo "Seed $seed did not meet validation selection. Evaluating its final step for diagnosis."
        candidates+=("$folder/step-$steps.pt")
    fi
done
"$python" -m kxvo.xfeat_eval --manifest "$dataset/manifest.csv" --checkpoint "${candidates[@]}" \
    --device "$device" --output "$output/held-out.json"
