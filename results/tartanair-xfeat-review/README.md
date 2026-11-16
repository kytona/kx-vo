# XFeat training and benchmark correction

The [500-step release](https://github.com/kytona/kx-vo/releases/tag/tartanair-xfeat-v1) measured a regression for its particular recipe. The frozen reliability head inherited changing input features, detector negatives included valid teacher detections, and the clean and blurred validation passes drew different warps. These defects confound conclusions about fine-tuning from those weights.

The corrected recipe used the same 1,000 training and 200 validation images and ran 1,500 steps per seed. It trains reliability alongside descriptors, uses all teacher detections to set occupied cells, and changes only blur between paired validation images. Selection requires at least one percentage point of blur recall gain, at most two points of clean recall loss, and no drop in blurred spatial correspondence coverage. Both corrected seeds missed the blur gain threshold. The Abandoned School and MegaDepth test sets were left out of the corrected training comparison. [Corrected training and replay metrics](metrics.json)

| Training seed | Baseline blur recall | Step 1,500 blur recall | Baseline clean recall | Step 1,500 clean recall |
| --- | ---: | ---: | ---: | ---: |
| 1 | 27.7% | 28.5% | 39.9% | 38.3% |
| 2 | 27.7% | 28.4% | 39.9% | 38.2% |

The primary trajectory metric is median horizontal relative translation error over continuous 5 s windows. The table gives the median and range across three generated-room fixtures with different texture and sensor seeds. Learned tracker inference used MPS at 640 px. Latency is per-frame estimator processing in offline replay.

| Baseline | Clean 5 s error | Blur 5 s error | Blur processing p95 |
| --- | ---: | ---: | ---: |
| Stock KLT | 0.085 m (0.050–0.151) | 0.059 m (0.036–0.151) | 4.70 ms (4.58–4.95) |
| Hybrid ORB | 0.065 m (0.059–0.123) | 0.052 m (0.047–0.137) | 11.31 ms (11.02–11.68) |
| Pretrained XFeat | 0.132 m (0.093–0.163) | 0.223 m (0.076–0.259) | 17.94 ms (17.35–18.69) |

On one separate synthetic development fixture (seed 3), the final corrected checkpoints produced blurred 5 s error of 0.076 m and 0.050 m, against 0.076 m for pretrained XFeat, 0.036 m for KLT and 0.047 m for ORB. This diagnostic did not change checkpoint selection.

On the original seed-1 blur trace, 5 s relative error was 0.151 m for KLT, 0.137 m for ORB, 0.223 m for pretrained XFeat, and 0.180 m and 0.223 m for the original two fine-tunes. RMS described global drift and ranked the arms differently. The original report's five SITL summaries matched their raw DataFlash evidence. Those runs exercise brief ExternalNav operation before an optical-flow fallback, using a fixed minimum blur floor. The box flight with motion-conditioned blur, repeated SITL acceptance, JSON lockstep, and hardware validation remain open.

The downloader's optional geometry mode was verified on six real TartanAir pairs across two routes. It retains adjacent RGB, depth, poses and camera intrinsics for the separate geometric-label study. The `kxbench quickstart` command generates a synthetic recording and a JSON scenario locally, then scores the recorded odometry track.
