#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 TRAINING_RUN [cpu|mps|cuda] [all|matches|replay|sitl]" >&2
    exit 2
fi
root=$1
device=${2:-cpu}
stage=${3:-all}
py=${KXVO_PYTHON:-.venv/bin/python}
dataset=runs/datasets/tartanair-xfeat
megadepth=${KXVO_MEGADEPTH:-runs/datasets/Mega1500}
arducopter=${KXVO_ARDUCOPTER:-runs/ardupilot/build/sitl/bin/arducopter}
dest=${KXVO_EVAL_OUTPUT:-"$root/evaluation"}
base=runs/xfeat-upstream/weights/xfeat.pt
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

case "$stage" in all|matches|replay|sitl) ;; *) echo "Unknown stage: $stage" >&2; exit 2;; esac
if [[ "$stage" == all || "$stage" == matches ]]; then
    [[ -d "$megadepth/megadepth_test_1500" ]] || { echo "MegaDepth-1500 missing in $megadepth" >&2; exit 2; }
fi
if [[ "$stage" == all || "$stage" == sitl ]]; then
    [[ -x "$arducopter" ]] || { echo "ArduCopter SITL binary missing: $arducopter" >&2; exit 2; }
fi

models=("$base")
names=(pretrained)
for seed in 1 2; do
    folder="$root/seed-$seed"
    # Only completed training runs can enter evaluation.
    name=$("$py" - "$folder" "${KXVO_DIAGNOSTIC_FINAL:-0}" <<'PY'
import json
from pathlib import Path
import sys

folder = Path(sys.argv[1])
config = json.loads((folder / "provenance.json").read_text())["config"]
steps = config["steps"]
last = json.loads((folder / "validation.jsonl").read_text().splitlines()[-1])
if last["step"] != steps:
    raise SystemExit(f"Incomplete training: {folder}")
path = folder / "selected.json"
if path.exists():
    print(json.loads(path.read_text())["checkpoint"])
elif sys.argv[2] == "1":
    print(f"step-{steps}.pt")
PY
)
    if [[ -n "$name" ]]; then
        names+=("seed-$seed")
        models+=("$folder/$name")
    fi
done
mkdir -p "$dest"

if [[ "$stage" == all || "$stage" == matches ]]; then
    "$py" -m kxvo.xfeat_eval --manifest "$dataset/manifest.csv" --checkpoint "${models[@]}" \
        --manifest-sha256 "$(cat experiments/tartanair-xfeat.sha256)" --device "$device" --output "$dest/held-out.json"
    for i in "${!models[@]}"; do
        "$py" -m kxvo.megadepth --dataset "$megadepth" --checkpoint "${models[i]}" \
            --seed 1 2 3 --device "$device" --output "$dest/megadepth-${names[i]}.json"
    done
fi

# Replay seeds vary both room textures and sensor noise.
if [[ "$stage" == all || "$stage" == replay ]]; then
    for seed in ${KXVO_SCENARIO_SEEDS:-1 2 3 4 5}; do
        fixture="$dest/fixture-$seed"
        if [[ ! -e "$fixture" ]]; then
            "$py" scripts/make-smoke-data.py "$fixture" --width 640 \
                --scene-seed "$((11 + seed))" --sensor-seed "$((7 + seed))"
        fi
        for condition in clean blur; do
            out="$dest/replay-$seed/$condition"
            KXVO_REPLAY_SEED="$seed" KXVO_DEVICE="$device" KXVO_WIDTH=640 KXVO_RANGE_OFFSET_Z=0 \
                KXVO_PYTHON="$py" scripts/replay-experiment.sh "$fixture/camera.json" \
                "$fixture/$condition" "$out" "${models[@]:1}"
            "$py" -m kxvo.score --reference "$fixture/$condition/log.jsonl" \
                --output "$out/metrics.json" "$out/"*.jsonl
        done
    done
fi

if [[ "$stage" == all || "$stage" == sitl ]]; then
    for seed in ${KXVO_SCENARIO_SEEDS:-1 2 3}; do
        for arm in klt orb "${names[@]}"; do
            tracker=(--tracker "$arm")
            for i in "${!names[@]}"; do
                if [[ "$arm" == "${names[i]}" ]]; then
                    tracker=(--tracker xfeat --checkpoint "${models[i]}")
                fi
            done
            out="$dest/sitl-$seed-$arm"
            rc=0
            "$py" -m kxvo.sitl --arducopter "$arducopter" --output "$out" "${tracker[@]}" \
                --seed "$seed" --delay-ms 80 --jitter-ms 15 --drop 0.1 --exposure-ms 12 \
                --blur-min-px 7 --blackout-seconds 1 --downlink-bps 1250 --fallback-source flow || rc=$?
            # Acceptance failures have a summary. Infrastructure errors abort.
            if [[ $rc -ne 0 && ! -s "$out/summary.json" ]]; then
                exit "$rc"
            fi
        done
    done
fi
