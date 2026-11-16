#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 OUTPUT [cpu|mps|cuda] [STEPS]" >&2
    exit 2
fi
output=$1
device=${2:-cpu}
steps=${3:-10000}
python=${KXVO_PYTHON:-.venv/bin/python}
dataset=runs/datasets/tartanair-xfeat
checkpoint=runs/xfeat-upstream/weights/xfeat.pt
manifest_hash=$(cat experiments/tartanair-xfeat.sha256)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

scripts/setup-xfeat.sh
if [[ ! -f "$dataset/manifest.csv" ]]; then
    "$python" -m kxvo.tartanair --plan experiments/tartanair-xfeat.json --output "$dataset"
fi
scenarios=()
for seed in 21 22 23 24 25; do
    fixture="$output/dev-$seed"
    "$python" - "$fixture" "$seed" <<'PYFIX'
from pathlib import Path
import sys
from kxvo.scenario import make_fixture
root, seed = Path(sys.argv[1]), int(sys.argv[2])
if not (root / "scenario.json").is_file():
    make_fixture(root, seed, seed + 10, degraded=True)
PYFIX
    scenarios+=(--validation-scenario "$fixture/scenario.json")
done
"$python" -m kxvo.train --manifest "$dataset/manifest.csv" --output "$output" \
    --checkpoint "$checkpoint" --device "$device" --width 320 --height 320 \
    --train-detector --manifest-sha256 "$manifest_hash" --seeds 1 2 --steps "$steps" \
    --validate-every 2500 --select-on-trajectory "${scenarios[@]}"
