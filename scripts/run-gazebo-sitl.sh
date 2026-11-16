#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 OUTPUT [kxvo.sitl arguments]" >&2
    exit 2
fi
root=$1
shift
plugin=${KXVO_GAZEBO_PLUGIN:-runs/ardupilot-gazebo}
build=${KXVO_GAZEBO_BUILD:-"$plugin/build"}
binary=${KXVO_ARDUCOPTER:-runs/ardupilot/build/sitl/bin/arducopter}
python=${KXVO_PYTHON:-.venv/bin/python}
[[ -f "$build/libArduPilotPlugin.so" || -f "$build/libArduPilotPlugin.dylib" ]] || {
  echo "Build ArduPilotPlugin from the configured Gazebo plugin checkout: $build" >&2
  exit 2
}
mkdir -p "$root"
build=$(cd "$build" && pwd)
plugin=$(cd "$plugin" && pwd)
export GZ_SIM_SYSTEM_PLUGIN_PATH="$build${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"
export GZ_SIM_RESOURCE_PATH="$plugin/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
export GZ_PARTITION="kxvo-$$"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
world=sim/iris-room.sdf
extra=(--simulated-plane-range)
relay=
if [[ -n "${KXVO_GAZEBO_CAMERA_PYTHON:-}" ]]; then
    world="$root/camera-world.sdf"
    "$python" scripts/texture-gazebo-room.py sim/iris-camera-room.sdf "$world"
    "$KXVO_GAZEBO_CAMERA_PYTHON" -s scripts/probe-gazebo-camera.py "$root/frame.rgb" --relay > "$root/camera.log" 2>&1 &
    relay=$!
    extra+=(--gazebo-camera-frame "$(cd "$root" && pwd)/frame.rgb")
fi
gz sim -v 3 -s -r "$world" > "$root/gazebo.log" 2>&1 &
gazebo=$!
trap 'if [[ -n "$relay" ]]; then kill "$relay" 2>/dev/null || true; wait "$relay" 2>/dev/null || true; fi; kill "$gazebo" 2>/dev/null || true; wait "$gazebo" 2>/dev/null || true' EXIT
"$python" -m kxvo.sitl --arducopter "$binary" --output "$root/flight" --instance 4 \
    --json-physics-port 9002 "${extra[@]}" "$@"
