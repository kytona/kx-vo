# Paired replay and simulator timing

## Five paired CPU replays

The scorer reports the median 5 s horizontal relative translation error over the windows every arm could score. It also counts each arm's lost windows against the eligible set. The table summarizes the median across five independently seeded room textures and sensor-noise realizations. Each fixture uses the same trajectory family and decay limits.

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
