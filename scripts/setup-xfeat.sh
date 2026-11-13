#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
target=runs/xfeat-upstream
revision=e92685f57f8318b18725c5c8c0bd28c7fe188d9a
if [[ ! -d "$target/.git" ]]; then
    git clone https://github.com/verlab/accelerated_features.git "$target"
    git -C "$target" checkout --detach "$revision"
fi
[[ "$(git -C "$target" rev-parse HEAD)" == "$revision" ]]
echo '0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b  runs/xfeat-upstream/weights/xfeat.pt' | shasum -a 256 -c -
