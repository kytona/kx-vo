#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
target=runs/ardupilot-gazebo
revision=f116b03552d05c8d0b0bed5ed67033a1183121d1
if [[ ! -d "$target/.git" ]]; then
    git init "$target"
    git -C "$target" fetch --depth 1 https://github.com/ArduPilot/ardupilot_gazebo.git "$revision"
    git -C "$target" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$target" rev-parse HEAD)" == "$revision" ]]
cmake_options=(-DCMAKE_BUILD_TYPE=Release)
if [[ "$(uname -s)" == Darwin ]]; then
    qt5=$(brew --prefix qt@5)
    qt_include="$PWD/$target/kxvo-qt.cmake"
    cat > "$qt_include" <<'EOF'
find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets Quick Qml QuickControls2)
EOF
    cmake_options+=("-DCMAKE_PREFIX_PATH=$qt5" "-DCMAKE_PROJECT_INCLUDE=$qt_include")
    if [[ "$qt5" == /opt/homebrew/* ]]; then
        cmake_options+=(-DCMAKE_OSX_ARCHITECTURES=arm64)
    fi
fi
GZ_VERSION=harmonic cmake -S "$target" -B "$target/build" "${cmake_options[@]}"
cmake --build "$target/build" --target ArduPilotPlugin -j "${KXVO_BUILD_JOBS:-3}"
