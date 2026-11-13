# XFeat experiment results

These are synthetic smoke and software-in-the-loop results. The training smoke used four synthetic training frames and four synthetic validation frames, two seeds and three optimiser steps per seed. The held-out trajectory used a different room texture seed and a 24 s trajectory. The blur clip added three artificial bursts.

The same clips were scored with stock KLT, ORB, pretrained XFeat and both smoke checkpoints. Stock KLT was substantially better on these trajectories. The smoke experiment proves that weights can be trained, validated and reloaded. It provides no evidence to select a learned tracker for flight.

The ArduCopter tests start on simulated GPS and switch to ExternalNav. The summary includes accepted vision inputs, the selected EKF source set, horizontal innovation records and landing. Raw DataFlash and frame logs stay in the local `runs/` directory.

- [Clean replay metrics](synthetic-clean.json) and [trajectory plot](synthetic-clean.png)
- [Blur replay metrics](synthetic-blur.json) and [trajectory plot](synthetic-blur.png)
- [Training smoke](training-smoke.json)
- [SITL summaries](sitl.json)
- [MegaDepth-1500 relative pose](megadepth.json) for the original weights and both smoke checkpoints

## Corrected harness and replay

The original SITL artifact above is retained for diagnosis. Its renderer used float32 latitude and its pass condition did not enforce a useful pose stream or stable hold. Use the corrected run summaries for conclusions. They include clean isolated runs, failures under concurrent CPU load, an optical-flow fallback case, and a five-metre pose injection with EKF innovation-gate evidence.

- [Corrected SITL summaries](review-sitl.json)
- [640 px clean comparison](review-clean-640.json) and [plot](review-clean-640.png)
- [640 px blur comparison](review-blur-640.json) and [plot](review-blur-640.png)

At 640 px, pretrained XFeat with KLT tracking and descriptor recovery scored 0.103 m horizontal RMS on the clean synthetic clip and 0.248 m on the artificially blurred clip. Stock KLT scored 0.298 m and 0.317 m. Both detector-training smoke checkpoints scored worse than stock in the same replay. These results describe this generated room and trajectory.
- [Detector-training smoke metrics](review-detector-smoke.json)

- [Pretrained MegaDepth-1500 PoseLib seed variation](review-megadepth-seeds.json)
