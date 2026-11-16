# TartanAir V2 XFeat fine-tuning checkpoints

These are two experimental sparse XFeat checkpoints for reproducing a visual odometry study. Both start from the official XFeat weights at revision `e92685f57f8318b18725c5c8c0bd28c7fe188d9a`. Kytona fine-tuned the descriptor backbone and detector for 500 steps per seed on public TartanAir V2 front-camera images with known image warps and directional motion blur. The pinned manifest records the 1,000 training frames and separate validation and test routes. [Training configuration and metrics](training.json) · [Source manifest](manifest.csv.gz)

Both checkpoints passed one native ArduPilot SITL fallback case. Both regressed against pretrained XFeat on the held-out TartanAir school, MegaDepth-1500, and clean and blurred room replays. Pretrained XFeat remains the recommended reference for this experiment. These checkpoints have simulation and replay evidence only. [Correspondence scores](held-out.json) · [SITL evidence](sitl.json)

The [review correction](../tartanair-xfeat-review/README.md) documents trainer defects that confound attribution of the original regression.

The files are PyTorch state dictionaries for the pinned XFeat architecture. Use them with the repository’s `--tracker xfeat --checkpoint PATH` option or the matching evaluator. Verify the download with `shasum -a 256 -c SHA256SUMS`.

| Release asset | SHA-256 |
| --- | --- |
| `xfeat-tartanair-v2-seed-1.pt` | `f96dba93e5cfaa4ae47b055ea130956339a9cba02b2ca7d317cecfc6588912a9` |
| `xfeat-tartanair-v2-seed-2.pt` | `7e3a696ece15330b4ce1c9b90df22574850200fa0330156a760b61c06b4c3be3` |

The weights incorporate and modify XFeat by VeRLab under Apache-2.0. Training used a subset of TartanAir V2 by the AirLab team at Carnegie Mellon University under CC BY 4.0. The source images were resized, converted to grayscale, warped and blurred during training. The release includes the checkpoint changes, configuration and attribution. Fetch the source images separately using the pinned downloader. [Upstream XFeat license](https://github.com/verlab/accelerated_features/blob/e92685f57f8318b18725c5c8c0bd28c7fe188d9a/LICENSE) · [TartanAir V2](https://tartanair.org/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
