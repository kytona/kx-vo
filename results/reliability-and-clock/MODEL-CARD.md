# XFeat TartanAir reliability-distillation checkpoints

These experimental sparse XFeat weights were fine-tuned by Kytona from VeRLab's pretrained checkpoint. Both seeds trained for 10,000 steps on 1,000 public TartanAir V2 source images. The architecture is unchanged. Training updates the descriptor backbone, detector and reliability head. Clean teacher reliability maps and detections supervise augmented image pairs.

Checkpoint selection used five synthetic development fixtures with shared continuous 5 s windows, lost-window accounting and a clean feature-recall check. Seed 1 selected step 5,000. Seed 2 selected step 10,000. [Training configuration and metrics](distillation-training.json)

| Checkpoint | School clean / blurred recall | Mean MegaDepth AUC @ 5° / 10° / 20° |
| --- | --- | --- |
| Pretrained XFeat | 41.2% / 30.5% | 43.18 / 56.85 / 67.97 |
| Distilled seed 1 | 44.2% / 34.6% | 42.68 / 56.08 / 66.51 |
| Distilled seed 2 | 43.7% / 33.6% | 42.04 / 55.18 / 65.87 |

Seed 1 passed the two-point MegaDepth tolerance at each threshold. Seed 2 exceeded it by 0.10 points at 20°. On five separate synthetic test fixtures, both improved relative translation error over pretrained XFeat. Stock KLT retained the lowest median blurred error. These checkpoints have held-out matching and offline replay evidence. Qualification of these specific weights in SITL and on FlyTimes remains pending. [Complete results](README.md)

## Download and use

Download the weights and checksums from the [distillation release](https://github.com/kytona/kx-vo/releases/tag/tartanair-xfeat-distilled-v2). Verify them in the download directory:

```sh
shasum -a 256 -c SHA256SUMS
```

The files are PyTorch state dictionaries for upstream XFeat revision `e92685f57f8318b18725c5c8c0bd28c7fe188d9a`. After installing the repository's `xfeat` extra and running `scripts/setup-xfeat.sh`, load a file with `kxvo replay --tracker xfeat --checkpoint PATH` alongside your camera calibration and recording arguments.

| Release asset | SHA-256 |
| --- | --- |
| `xfeat-tartanair-distilled-seed-1.pt` | `3f2379c2a0ea422fdc74d2dd0dd420e3d9db353a2f5784a6114ea92e4ca533f4` |
| `xfeat-tartanair-distilled-seed-2.pt` | `0c96ac3b4ba3f8d445413b4f44164e6b82546b89f68e97fcdb5a2381f6a2a170` |

## Attribution

The weights incorporate and modify XFeat by VeRLab under Apache-2.0. Training used TartanAir V2 by Carnegie Mellon University's AirLab under CC BY 4.0. Source images were resized, converted to grayscale, warped and blurred. The release supplies the upstream licence and attribution. [XFeat source and licence](https://github.com/verlab/accelerated_features/tree/e92685f57f8318b18725c5c8c0bd28c7fe188d9a) · [TartanAir V2](https://tartanair.org/) · [Dataset licence](https://creativecommons.org/licenses/by/4.0/)
