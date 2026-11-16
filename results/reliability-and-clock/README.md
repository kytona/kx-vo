# Paired replay and simulator timing

## XFeat reliability distillation

The revised trainer matches reliability to the frozen teacher’s clean maps. It projects the second view’s targets by inverse homography and masks cells outside the original image. The fixed recipe verifies the published manifest hash before training and checks five paired development replays for each checkpoint.

Both seeds completed 10,000 steps. The development criterion selected step 5,000 for seed 1 and step 10,000 for seed 2, with 49.3% and 33.7% median relative error improvements over pretrained XFeat on their paired development windows. On 20 sampled validation frames, the pretrained reliability mean was 0.213 and the previous recipe’s first seed averaged 0.986. The selected checkpoints averaged 0.214 and 0.207. Their mean spatial standard deviations were 0.117 and 0.113, compared with 0.125 for pretrained and 0.036 for the previous recipe. [Training, checkpoint hashes and reliability diagnostic](distillation-training.json)

| XFeat weights | Held-out school clean recall | Held-out school blur recall | MegaDepth AUC @ 5° / 10° / 20° |
| --- | ---: | ---: | ---: |
| Pretrained | 41.2% | 30.5% | 43.18 / 56.85 / 67.97 |
| Distilled seed 1 | 44.2% | 34.6% | 42.68 / 56.08 / 66.51 |
| Distilled seed 2 | 43.7% | 33.6% | 42.04 / 55.18 / 65.87 |

MegaDepth values average three paired PoseLib seeds. Seed 1 cleared the two-point regression tolerance at all thresholds. Seed 2 exceeded it by 0.10 points at 20°. [Held-out school](distillation-held-out.json) · [MegaDepth](distillation-megadepth.json)

| Tracker | Clean 5 s error | Blur 5 s error | Blurred windows lost of 2,545 |
| --- | ---: | ---: | ---: |
| KLT | 0.0713 m | 0.0515 m | 0 |
| Hybrid ORB | 0.0709 m | 0.0625 m | 89 |
| Hybrid pretrained XFeat | 0.1130 m | 0.1661 m | 293 |
| Hybrid distilled seed 1 | 0.0823 m | 0.0879 m | 0 |
| Hybrid distilled seed 2 | 0.0652 m | 0.0534 m | 0 |

The table uses the median over five separate synthetic test fixtures, with each per-fixture error computed on windows supported by all five arms. Both distilled checkpoints improved on pretrained XFeat in this comparison. Stock KLT had the lowest median blurred error. The replay and matching results support seed 1 as the stronger learned candidate for further testing. A tuned checkpoint has yet to be qualified in this study’s SITL and aircraft profiles. [Shared windows, losses and scores](distillation-replay.json)

## Five paired CPU replays

The scorer reports the median 5 s horizontal relative translation error over the windows every arm could score. It also counts each arm's lost windows against the eligible set. The table summarizes the median across five independently seeded room textures and sensor-noise realizations. Each fixture uses the same trajectory family and scoring settings.

| Stock tracker | Clean median | Artificial blur median |
| --- | ---: | ---: |
| KLT | 0.072 m | 0.051 m |
| Hybrid ORB | 0.071 m | 0.063 m |
| Hybrid pretrained XFeat | 0.113 m | 0.166 m |

Three runs on the same blurred KLT fixture all measured 0.151279 m. Across the five textures and sensor seeds, KLT's blurred error ranged from 0.038 m to 0.151 m. The fixture-level bootstrap interval for ORB's paired blur difference from KLT spans zero. The five synthetic fixtures give limited precision for small improvements. The published blur stress uses artificial exposure and cannot establish the dominant failure mode of the FlyTimes capture chain. [Replay metrics and counts](paired-replay.json)

## ArduPilot clocks and pose inputs

The corrected JSON probe hashes complete received servo packets. All 2,000 packets matched across three runs without added load and three with two CPU workers. Its stationary model exercises protocol timing.

The native runner's optional boot-clock mode schedules fault and mission deadlines from ArduCopter’s `SYSTEM_TIME`. Six stock KLT runs with the same seed passed the fallback profile, including three with two extra CPU workers. The no-added-load trials had 0.063–0.135 m of hold drift. The added-load trials had 0.073–0.079 m. Other local experiments were active. The simulation still exchanges camera truth and commands through asynchronous MAVLink delivery. The results capture repeatability spread under those conditions. Drop accounting separated the 31 blackout drops per run, 51–60 injected random drops and 26–36 reordered frames. [Servo packet evidence](servo-clock.json) · [Same-seed fallback runs](boot-clock-fallback.json) · [MAVLink boot clock](https://mavlink.io/en/messages/common.html#SYSTEM_TIME)

The pose-stream scenario also passed with caller-supplied JSONL extracted from an earlier simulated flight. The scorecard identifies the external file by hash and includes DataFlash fusion evidence. [Recorded-pose scorecard](user-pose-stream.json)
