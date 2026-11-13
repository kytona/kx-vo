#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
python_bin="$(command -v "${KXVO_PYTHON:-python3}")"
python_bin="$(cd "$(dirname "$python_bin")" && pwd)/$(basename "$python_bin")"
target=runs/ardupilot
revision=dbe792162d06cab66c3475fd5556bf7a120f119e
if [[ ! -d "$target/.git" ]]; then
    git clone --depth 1 --branch Copter-4.7.1 https://github.com/ArduPilot/ardupilot.git "$target"
fi
git -C "$target" diff --quiet
[[ "$(git -C "$target" rev-parse HEAD)" == "$revision" ]]
git -C "$target" submodule update --init --recursive -j 8
cd "$target"
"$python_bin" ./waf configure --board sitl
"$python_bin" ./waf copter -j "${KXVO_BUILD_JOBS:-8}"
