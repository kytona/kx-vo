# kx-vo

Range-aided monocular visual odometry for small indoor drones, with a replay and ArduPilot benchmark. The estimator uses a forward-facing camera, downward range and flight-controller attitude to produce a local pose. It can publish that pose to ArduPilot ExternalNav over MAVLink.

- **`kxvo`** calibrates cameras, records flights, replays recordings and estimates pose using KLT, ORB or XFeat.
- **`kxbench`** runs replay and pose-stream scenarios and writes a scorecard with thresholds, measured results and a verdict.

Native ArduPilot SITL and the Gazebo RGB profile have passed the scenarios below. FlyTimes hardware validation is pending. [Detailed results and run evidence](results/README.md)

## Quickstart

Generate and score a synthetic flight on CPU. The demo needs Python 3.10 or later, with Python 3.11 used for the published runs.

```sh
git clone https://github.com/kytona/kx-vo.git
cd kx-vo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
kxbench quickstart runs/demo
```

The demo writes the generated recording to `runs/demo/capture/`, the scenario to `runs/demo/scenario.json`, and the scorecard to `runs/demo/result/score.json`. The scorecard reports pose coverage, horizontal and yaw error, and median relative translation error over continuous 5 s windows.

```sh
kxbench run runs/demo/scenario.json --output runs/demo-repeat --tracker orb
```

Use a fresh output directory for each run. Exit status `0` means the run met its thresholds, `1` means it failed a threshold, and `2` means it could not be evaluated.

## What you can benchmark

| Your task | Entry point | Inputs and result |
| --- | --- | --- |
| Compare visual tracking models | `kxvo replay` and `kxvo.score` | Use the same recorded images, range and attitude for every tracker. Compare error on shared windows, outages and processing time. |
| Check a repeatable odometry scenario | `kxbench run` | Supply a JSON manifest with recording, calibration, reference and limits. Get a scorecard and pose trace. |
| Debug an external navigation integration | `kxbench run` with `pose_stream` | Supply your own timestamped NED poses or a built-in fault fixture. Inspect publication and flight-controller fusion in DataFlash. |
| Exercise recovery and guidance | `python -m kxvo.sitl` | Run native ArduPilot with delayed frames, telemetry budgets, blackouts, outliers or the box route. |
| Evaluate a Gazebo camera pipeline | `scripts/run-gazebo-sitl.sh` | Run the pinned Iris physics model and the optional mounted RGB camera. Retain the effective configuration and acceptance evidence. |

Bring a model into the visual estimator through the tracker implementation in [kxvo/trackers.py](kxvo/trackers.py). For a model that already produces positions, the pose-stream contract below lets you test its output directly. PX4 adapters, VLM mission policies and additional environments are future extensions.

## Measured results

| Profile | Observed result |
| --- | --- |
| Native ArduPilot fallback | Stock KLT and pretrained XFeat each passed 3/3 runs. DataFlash confirmed the switch from ExternalNav to optical flow. |
| Native motion-blur box | Stock KLT reached all four waypoints of the 0.7 m box and landed. Simulated angular motion produced at least 3 px of blur on 334 frames. |
| Pose-stream injection | DataFlash confirmed rejection of a 5 m outlier. The deliberate stale stream failed acceptance. |
| ArduPilot JSON clock | The stationary fixture completed 2,000 exchanges per run. The received servo packets matched across three normal runs and three with two busy CPU workers. |
| Gazebo Harmonic RGB | The 15° downward camera profile passed clean ExternalNav and blackout-to-GPS recovery. Clean hold estimator RMS was 0.093 m horizontally. |
| XFeat fine-tuning | Both distilled seeds improved held-out school matching and synthetic trajectory error over pretrained XFeat. Seed 1 passed the general matching regression tolerance. |

The passing Gazebo profile uses emulated floor range and host receive times to associate camera frames with simulator truth. The report includes earlier camera configurations that exceeded the error limit. [Simulator and geometric training results](results/benchmark-followthrough/README.md)

On five blurred synthetic CPU replays at 640 px, the median 5 s relative translation error on shared windows was 0.051 m for KLT, 0.063 m for ORB and 0.166 m for pretrained XFeat. The scorer also reports the windows each tracker lost. [Paired replay results](results/reliability-and-clock/README.md)

## Run the simulator benchmarks

Install the SITL extras and build the pinned ArduCopter 4.7.1 binary. The build requires the native ArduPilot toolchain. [ArduPilot SITL setup](https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html)

```sh
python -m pip install -e '.[sitl]'
KXVO_PYTHON=.venv/bin/python scripts/build-sitl.sh

kxbench run experiments/pose-stream-outlier.json --output runs/pose-outlier
kxbench run experiments/pose-stream-stale.json --output runs/pose-stale
```

The stale scenario is expected to exit with status 1. The scorecard and flight directory retain the observations and DataFlash evidence.

For your own ExternalNav source, create a version 1 `pose_stream` scenario with a `seed` and a `pose_stream` JSONL path relative to the manifest. Each line supplies elapsed seconds in `t`, position `p` in metres NED relative to the start of the simulated translation, and roll, pitch and yaw `rpy` in radians. Optional fields are `valid`, `sigma_m` and `reset`. Input times must increase. Stale and invalid samples stop publication.

```json
{"version":1,"track":"pose_stream","seed":1,"pose_stream":"poses.jsonl"}
```

Run the box and stationary clock probe with the same binary:

```sh
python -m kxvo.sitl --arducopter runs/ardupilot/build/sitl/bin/arducopter \
  --output runs/box --box-m 0.7 --exposure-ms 55 --timeout 125

python -m kxvo.lockstep --arducopter runs/ardupilot/build/sitl/bin/arducopter \
  --output runs/json-clock
```

To run the native optical-flow fallback profile, use the same injected link settings across seeds. The documented results use seeds 1 through 3.

```sh
python -m kxvo.sitl --arducopter runs/ardupilot/build/sitl/bin/arducopter \
  --output runs/fallback-1 --seed 1 --delay-ms 80 --jitter-ms 15 --drop 0.1 \
  --exposure-ms 12 --blur-min-px 7 --blackout-seconds 1 --downlink-bps 1250 \
  --fallback-source flow
```

For the pretrained arm, install the `xfeat` extra, run `scripts/setup-xfeat.sh`, and add `--tracker xfeat --checkpoint runs/xfeat-upstream/weights/xfeat.pt`. Run the cases serially so they use the same host resources.

For native fallback timing, `--clock autopilot` schedules against ArduCopter's MAVLink boot clock. `scripts/repeat-sitl-clock.sh` runs the same seed three times without added load and three times with two CPU workers. The runner reports injected drops, blackout losses, reordering, queue overflow and stale telemetry separately. Camera truth still arrives through MAVLink. [MAVLink system time](https://mavlink.io/en/messages/common.html#SYSTEM_TIME)

For Gazebo Harmonic, install the platform dependencies and build the pinned ArduPilot plugin. The default launcher uses Gazebo physics with kx-vo’s synthetic renderer. To use the mounted Gazebo RGB camera, set `KXVO_GAZEBO_CAMERA_PYTHON` to an interpreter with the matching `gz.transport13`, `gz.msgs10` and Protobuf packages. [Gazebo setup](https://ardupilot.org/dev/docs/sitl-with-gazebo.html)

```sh
scripts/build-gazebo-plugin.sh
scripts/run-gazebo-sitl.sh runs/gazebo-clean --timeout 140
scripts/run-gazebo-sitl.sh runs/gazebo-recovery --timeout 140 \
  --blackout-seconds 1 --delay-ms 50 --fallback-source gps
```

## Fine-tuning XFeat

We started with VeRLab's pretrained sparse XFeat and kept its architecture. Fine-tuning updates the descriptor backbone, detector and reliability head. The hybrid estimator uses learned detections and descriptors to seed and recover tracks, with KLT supplying frame-to-frame continuity.

The experiments progressed through four stages:

1. **Warp and blur training.** We trained two independent seeds on 1,000 public TartanAir source images, with 200 validation images on separate routes and a school environment held out for testing.
2. **Trainer diagnostics.** We measured feature recall, spatial coverage and reliability maps, then corrected detector negatives, paired validation warps and the reliability objective. These checks exposed why an early recipe regressed.
3. **Geometric supervision.** We downloaded neighboring RGB frames, depth and poses. Bidirectional projection and depth checks reject occluded correspondences. Two geometric seeds preserved more real-viewpoint matching than their warp-trained controls.
4. **Teacher distillation and trajectory selection.** We distilled clean reliability maps from the frozen pretrained model and trained both seeds for 10,000 steps. Five development replays determined checkpoint selection. Separate matching and trajectory benchmarks measured generalization.

The selected checkpoints were step 5,000 for seed 1 and step 10,000 for seed 2. They improved on pretrained XFeat in the held-out school and separate synthetic replays.

| Model | School clean recall | School blurred recall | Synthetic blurred 5 s error |
| --- | ---: | ---: | ---: |
| Pretrained XFeat | 41.2% | 30.5% | 0.1661 m |
| Distilled XFeat, seed 1 | 44.2% | 34.6% | 0.0879 m |
| Distilled XFeat, seed 2 | 43.7% | 33.6% | 0.0534 m |
| Classical KLT | | | 0.0515 m |

The trajectory column is the median across five synthetic fixtures on windows shared by all arms. The scorer separately reports lost windows. Seed 1 stayed within the two-point MegaDepth regression tolerance. Seed 2 exceeded it by 0.10 points at 20°. These tuned checkpoints have held-out matching and offline replay evidence. Their SITL and FlyTimes qualification is pending. [Training, comparison counts and generalization results](results/reliability-and-clock/README.md)

### Download and try the trained weights

Install the XFeat and plotting extras, then download the selected checkpoints. The release includes the model card, source attribution and checksums.

```sh
python -m pip install -e '.[xfeat,plots]'
scripts/setup-xfeat.sh
mkdir -p runs/distilled
gh release download tartanair-xfeat-distilled-v2 --repo kytona/kx-vo \
  --dir runs/distilled --pattern '*.pt' --pattern SHA256SUMS
(cd runs/distilled && shasum -a 256 -c SHA256SUMS)

kxbench run runs/demo/scenario.json --output runs/demo-distilled \
  --tracker xfeat --device mps \
  --checkpoint runs/distilled/xfeat-tartanair-distilled-seed-1.pt
```

Use `cpu` or `cuda` for your device. The download command uses the GitHub CLI. You can also download the assets from the release page. [Selected weights](https://github.com/kytona/kx-vo/releases/tag/tartanair-xfeat-distilled-v2) · [Model card](results/reliability-and-clock/MODEL-CARD.md)

Compare all four trackers on the quickstart capture and generate a trajectory plot:

```sh
KXVO_DEVICE=mps KXVO_RANGE_OFFSET_Z=0 scripts/replay-experiment.sh \
  runs/demo/camera.json runs/demo/capture runs/demo-comparison \
  runs/distilled/xfeat-tartanair-distilled-seed-1.pt
python -m kxvo.score --reference runs/demo/capture/log.jsonl \
  --output runs/demo-comparison/metrics.json --plot runs/demo-comparison/trajectory.png \
  runs/demo-comparison/*.jsonl
```

The comparison contains KLT, ORB, pretrained XFeat and the downloaded checkpoint. Its `paired_relative_windows` field reports errors on common windows and the windows lost by each arm.

### Reproduce the training

```sh
python -m pip install -e '.[xfeat,benchmark]'
scripts/run-tartanair-xfeat.sh runs/my-tartanair mps
scripts/run-geometric-xfeat.sh runs/my-geometric mps
```

Use `cpu` or `cuda` for the corresponding device. Both scripts start two training seeds from pinned pretrained XFeat weights. The geometric experiment also runs two warp-trained controls. Its labels use neighboring TartanAir RGB frames, depth and camera poses with bidirectional visibility checks. The downloader fetches selected members of the pinned archives and resumes an interrupted download with the same command. [RGB route plan](experiments/tartanair-xfeat.json) · [Geometric route plan](experiments/tartanair-geometry.json) · [TartanAir V2](https://tartanair.org/)

The RGB recipe verifies the frozen manifest hash, distils reliability from the pretrained teacher, and runs 10,000 steps per seed by default. It evaluates five synthetic development replays at each checkpoint. Selection requires a median improvement of at least 15% on shared 5 s windows, an improvement on every fixture, preserved window coverage and a clean recall check. The metrics also report windows each arm lost. Run final evaluation explicitly after freezing the checkpoints. The stage can be `matches`, `replay`, `sitl` or `all`. The matches stage requires MegaDepth-1500 and the SITL stage requires a local ArduCopter build. [MegaDepth setup](https://github.com/verlab/accelerated_features#evaluation)

```sh
scripts/evaluate-tartanair-xfeat.sh runs/my-tartanair mps replay
```

The first 500-step experimental weights remain available for reproducing the early recipe. Its release note documents the trainer defects and links to the corrections. [Checkpoint release](https://github.com/kytona/kx-vo/releases/tag/tartanair-xfeat-v1) · [Model card](results/tartanair-xfeat/MODEL-CARD.md) · [Review correction](results/tartanair-xfeat-review/README.md)

## Record and replay a real camera

The intended FlyTimes setup captures the HDZero goggles’ HDMI output on a ground computer and receives MAVLink `ATTITUDE` and `DISTANCE_SENSOR` from the flight controller. Calibrate through the same capture chain and measure video delay before comparing against a reference trajectory. Disable the goggles’ on-screen display or mask it in the calibration file.

```sh
kxvo calibrate --video 0 --board 9x6 --square 0.024 --fisheye --uptilt 20 \
  --no-show --out flytimes_cam.json
kxvo run --camera flytimes_cam.json --video 0 --mavlink udpin:0.0.0.0:14550 \
  --record runs/shadow1
kxvo replay --camera flytimes_cam.json runs/shadow1 \
  --output runs/shadow1/klt.jsonl
python -m kxvo.score --output runs/shadow1/health.json runs/shadow1/klt.jsonl
```

This workflow records in shadow mode. Set `--video-latency-ms` to the measured capture delay. Pose publication requires `--send`. The repository includes starting [ArduPilot parameters](params/ardupilot_kxvo.parm) and an experimental [Betaflight P3 profile](params/betaflight_kx_extnav.txt). A timestamped position and yaw reference is needed for metric error. Use `python -m kxvo.ground_truth --help` to export an ArUco-based reference, then supply it to `kxvo.score` with `--reference`. Replay the same recording, camera and mount settings for each tracker.

## Estimator assumptions

The downward range anchors scale to a shared flat floor. Floor landmarks are checked across views, and other landmarks are triangulated as the camera moves. The estimator solves body pose from those tracked landmarks. It uses flight-controller attitude for initial leveling and range compensation. ORB and XFeat can seed tracks and re-associate lost landmarks by descriptor.

A glossy or poorly textured floor, steps, and obstacles under the rangefinder can compromise scale. Pose drift accumulates over distance without loop closure. The simulator measurements apply to the recorded profiles. Real HDZero latency, telemetry transport and receiver fusion still need measurement on the aircraft.

## Repository map

| Path | Contents |
| --- | --- |
| [kxvo/](kxvo/) | Estimator, calibration, replay, scoring, SITL and training |
| [experiments/](experiments/) | Dataset splits and pose-stream manifests |
| [sim/](sim/) | Gazebo worlds |
| [scripts/](scripts/) | Build, training and evaluation entry points |
| [results/](results/) | Scorecards, plots, provenance and model card |

[Apache-2.0 licence](LICENSE)
