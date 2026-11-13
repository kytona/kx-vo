# XFeat experiment results

These are synthetic smoke and software-in-the-loop results. The training smoke used four synthetic training frames and four synthetic validation frames, two seeds and three optimiser steps per seed. The held-out trajectory used a different room texture seed and a 24 s trajectory. The blur clip added three artificial bursts.

The same clips were scored with stock KLT, ORB, pretrained XFeat and both smoke checkpoints. Stock KLT was substantially better on these trajectories. The smoke experiment proves that weights can be trained, validated and reloaded. It provides no evidence to select a learned tracker for flight.

The ArduCopter tests start on simulated GPS and switch to ExternalNav. The summary includes accepted vision inputs, the selected EKF source set, horizontal innovation records and landing. Raw DataFlash and frame logs stay in the local `runs/` directory.

- [Clean replay metrics](synthetic-clean.json) and [trajectory plot](synthetic-clean.png)
- [Blur replay metrics](synthetic-blur.json) and [trajectory plot](synthetic-blur.png)
- [Training smoke](training-smoke.json)
- [SITL summaries](sitl.json)
- [MegaDepth-1500 relative pose](megadepth.json) for the original weights and both smoke checkpoints
