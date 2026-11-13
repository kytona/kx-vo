#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 4 ]]; then
    echo "Usage: $0 CAMERA_JSON RUN_DIR RESULTS_DIR TUNED_CHECKPOINT [TUNED_CHECKPOINT ...]" >&2
    exit 2
fi
camera=$1
run=$2
results=$3
shift 3
python_bin="${KXVO_PYTHON:-.venv/bin/python}"
width="${KXVO_WIDTH:-320}"
mkdir -p "$results"
for seed in "${KXVO_REPLAY_SEED:-1}"; do
    for tracker in klt orb; do
        "$python_bin" -m kxvo.cli replay --camera "$camera" --width "$width" \
            --tracker "$tracker" --seed "$seed" --output "$results/$tracker-$seed.jsonl" "$run"
    done
    index=0
    for checkpoint in runs/xfeat-upstream/weights/xfeat.pt "$@"; do
        "$python_bin" -m kxvo.cli replay --camera "$camera" --width "$width" \
            --tracker xfeat --checkpoint "$checkpoint" --seed "$seed" \
            --output "$results/xfeat-$index-$seed.jsonl" "$run"
        index=$((index + 1))
    done
done
