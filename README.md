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

## Measured results

| Profile | Observed result |
| --- | --- |
| Native ArduPilot fallback | Stock KLT and pretrained XFeat each passed 3/3 runs. DataFlash confirmed the switch from ExternalNav to optical flow. |
| Native motion-blur box | Stock KLT reached all four waypoints of the 0.7 m box and landed. Simulated angular motion produced at least 3 px of blur on 334 frames. |
| Pose-stream injection | DataFlash confirmed rejection of a 5 m outlier. The deliberate stale stream failed acceptance. |
| ArduPilot JSON clock | The stationary fixture produced identical 2,000-tick traces across three normal runs and three with two busy CPU workers. |
| Gazebo Harmonic RGB | The 15° downward camera profile passed clean ExternalNav and blackout-to-GPS recovery. Clean hold estimator RMS was 0.093 m horizontally. |
| XFeat fine-tuning | Depth-and-pose supervision beat the warp-trained controls on the same real-viewpoint validation pairs. Neither seed reached the blur-recall gain required for held-out testing. |

The passing Gazebo profile uses emulated floor range and host receive times to associate camera frames with simulator truth. The report includes earlier camera configurations that exceeded the error limit. [Simulator and geometric training results](results/benchmark-followthrough/README.md)

On three blurred synthetic replays at 640 px, the median 5 s relative translation error was 0.059 m for KLT, 0.052 m for ORB and 0.223 m for pretrained XFeat. XFeat’s median per-run p95 processing time on the Mac GPU was 17.9 ms in offline replay. [Replay metrics and ranges](results/tartanair-xfeat-review/README.md)

## Run the simulator benchmarks

Install the SITL extras and build the pinned ArduCopter 4.7.1 binary. The build requires the native ArduPilot toolchain. [ArduPilot SITL setup](https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html)

```sh
python -m pip install -e '.[sitl]'
KXVO_PYTHON=.venv/bin/python scripts/build-sitl.sh

kxbench run experiments/pose-stream-outlier.json --output runs/pose-outlier
kxbench run experiments/pose-stream-stale.json --output runs/pose-stale
```

The stale scenario is expected to exit with status 1. The scorecard and flight directory retain the observations and DataFlash evidence. Run the box and stationary clock probe with the same binary:

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

For Gazebo Harmonic, install the platform dependencies and build the pinned ArduPilot plugin. The default launcher uses Gazebo physics with kx-vo’s synthetic renderer. To use the mounted Gazebo RGB camera, set `KXVO_GAZEBO_CAMERA_PYTHON` to an interpreter with the matching `gz.transport13`, `gz.msgs10` and Protobuf packages. [Gazebo setup](https://ardupilot.org/dev/docs/sitl-with-gazebo.html)

```sh
scripts/build-gazebo-plugin.sh
scripts/run-gazebo-sitl.sh runs/gazebo-clean --timeout 140
scripts/run-gazebo-sitl.sh runs/gazebo-recovery --timeout 140 \
  --blackout-seconds 1 --delay-ms 50 --fallback-source gps
```

## Reproduce XFeat training

```sh
python -m pip install -e '.[xfeat,benchmark]'
scripts/run-tartanair-xfeat.sh runs/my-tartanair mps
scripts/run-geometric-xfeat.sh runs/my-geometric mps
```

Use `cpu` or `cuda` for the corresponding device. Both scripts start two training seeds from pinned pretrained XFeat weights. The geometric experiment also runs two warp-trained controls. Its labels use neighboring TartanAir RGB frames, depth and camera poses with bidirectional visibility checks. The downloader fetches selected members of the pinned archives and resumes an interrupted download with the same command. [RGB route plan](experiments/tartanair-xfeat.json) · [Geometric route plan](experiments/tartanair-geometry.json) · [TartanAir V2](https://tartanair.org/)

Training writes validation metrics, checkpoint hashes and provenance. Run final evaluation explicitly after freezing the checkpoints. The stage can be `matches`, `replay`, `sitl` or `all`. The matches stage requires MegaDepth-1500 and the SITL stage requires a local ArduCopter build. [MegaDepth setup](https://github.com/verlab/accelerated_features#evaluation)

```sh
scripts/evaluate-tartanair-xfeat.sh runs/my-tartanair mps replay
```

The original experimental weights are available for reproducing their results. The release note explains the trainer defects that confound the first run and links to the corrected recipe. [Checkpoint release](https://github.com/kytona/kx-vo/releases/tag/tartanair-xfeat-v1) · [Model card](results/tartanair-xfeat/MODEL-CARD.md) · [Review correction](results/tartanair-xfeat-review/README.md)

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
