# kxvo

Camera-only monocular visual odometry for small indoor drones, running on a ground computer. It reads the FPV video feed and the flight controller's rangefinder and attitude, and publishes the drone's position as MAVLink `VISION_POSITION_ESTIMATE` back to the flight controller.

Built for the Kytona FlyTimes 85 HDZero test drone (GOKU F405 HD, MTF-02P flow and range sensor). The output is intended for ArduPilot ExternalNav and the KX Betaflight P3 receiver.

**Status:** The synthetic benchmark and ArduCopter SITL have been exercised. Real FlyTimes capture, receiver fusion and flight remain pending. See [results](results/README.md) for the measured scope.

## How it works

Monocular VO cannot measure scale. kxvo takes scale from the floor:

1. The downward rangefinder gives camera height. Corners that look clearly below the horizon are placed on the floor plane in metres. A corner is accepted as floor only if the floor point also lines up with where the corner was first seen, from a camera position at least 8 cm away. That two-view check keeps wall corners from being treated as floor.
2. Corners on walls and furniture are triangulated once the drone has moved enough.
3. Each frame's pose is solved from this small map. Floor landmarks lead whenever there are at least 25 of them, so scale stays tied to the rangefinder. Pose solving is a guided, trimmed PnP from a constant-velocity prediction, refined with camera height held to the rangefinder reading.
4. Tracks are re-tracked against a keyframe every 8 frames to limit tracking drift.

The flight controller's attitude is used only to level the start pose and to turn the slant range into vertical height. This is loose use of the FC, not tightly coupled visual-inertial odometry.

kxvo publishes estimates only. The normal live estimator sends pose and telemetry requests. It stops pose publication when its estimate or required inputs are invalid or stale. Verify source switching and recovery on the chosen FC. The separate SITL runner sends flight commands to its own simulator process.

## Synthetic results

The room is a ray-cast textured box, 8 x 6 x 3 m. The camera is 320 x 240 with about 94° horizontal field of view and 15° uptilt, flown on a smooth loop at about 1 m height and 0.4 m/s. Simulated sensors: rangefinder noise 1 cm, attitude noise 0.3°, yaw drift 0.2°/s.

| Run | Path | Horizontal RMSE | End error | Yaw RMSE |
|---|---|---|---|---|
| One loop, 24 s, mean of 5 seeds | 8.6 m | 0.13 m (1.5% of path) | 0.34 m | 1.1° |
| Two loops, 48 s, mean of 3 seeds | 17.2 m | 0.60 m | 1.57 m (9.2% of path) | 3.5° |

Read these as a best case. The renderer has no motion blur, rolling shutter, lighting change, video compression, OSD overlay or real latency. Drift grows with distance because there is no loop closure, and yaw drift is the main error over two loops. Cancelling that drift is the job of map matching (AI-NAV-PLAN stage 5, `Claude_Workspace/Products/drones-kytona/kx-flytimes-build/AI-NAV-PLAN.md`).

Processing runs at about 275 frames per second at 320 x 240 on an Apple M5. That leaves plenty of headroom for a 60 fps HDZero feed.

```sh
uv venv .venv && uv pip install --python .venv/bin/python -e ".[test]"
.venv/bin/python -m pytest -q
.venv/bin/kxvo synth --seeds 5         # the benchmark above
```

## Hardware path

```
FlyTimes camera -> HDZero VTX -> BoxPro goggles -> mini HDMI out -> USB capture dongle -> Mac (kxvo)
Mac (kxvo) -> data link -> GOKU F405 UART -> ArduPilot VISO / KX Betaflight P3
GOKU F405 -> data link -> Mac: ATTITUDE, DISTANCE_SENSOR (MTF-02P)
```

**Data link without a Wi-Fi board.** At 10 Hz, compact unsigned MAVLink 2 poses can take about 440 B/s. A nonzero reset counter can keep the covariance extension on the wire, raising this to about 1,290 B/s before other traffic. Measure both cases.

- **ArduPilot:** ELRS MAVLink mode (ELRS 3.5+) on the existing receiver. ExpressLRS recommends it over Airport for MAVLink. Community reports put its throughput at about 1.25 kB/s. It needs an ESP-based transmitter module the Mac can reach over USB or the TX backpack. Check whether the RadioMaster Pocket's internal module qualifies before relying on it.
- **Both firmwares:** ELRS Airport, a transparent serial link. A second small ESP-based ELRS receiver goes on a spare GOKU UART and an ELRS TX module plugs into the Mac by USB. One community report puts throughput at about 0.5 kB/s, so keep 10 Hz compact or lower the rate.
- **Bench only:** USB to the flight controller.

Measure the real throughput on the bench before flying.

## Using it

1. **Turn off the goggle OSD on the HDMI feed**, or mask it. Static overlay text looks like features that never move. Add rectangles to `mask` in the camera JSON.
2. **Calibrate through the real video path**, at the processing width:
   ```sh
   kxvo calibrate --video 0 --board 9x6 --square 0.024 --fisheye --uptilt 20 --out flytimes_cam.json
   ```
3. **Measure the video latency** by filming a millisecond clock on the Mac screen and comparing it with the clock shown in the live feed. Pass it as `--video-latency-ms`, and put the total lag into `VISO_DELAY_MS`.
4. **Shadow mode first**: run with no `--send` and record. Compare the VO track against the overhead ground-truth camera (AI-NAV-PLAN section 5):
   ```sh
   kxvo run --camera flytimes_cam.json --video 0 --mavlink udpin:0.0.0.0:14550 --record runs/shadow1 --show
   kxvo replay --camera flytimes_cam.json runs/shadow1
   ```
5. **Only after shadow logs agree with ground truth:** load `params/ardupilot_kxvo.parm` (VO as EKF source set 2 on an RC switch, flow stays set 1) or `params/betaflight_kx_extnav.txt`, and add `--send`.

## Reproduce the XFeat experiment

From a fresh clone, create a Python 3.11 virtual environment and install `kxvo[test,xfeat,plots,benchmark,sitl]` in editable mode. The default live estimator only needs the base install. The tested simulator build was native macOS with Xcode Command Line Tools. Activate the environment with `source .venv/bin/activate` before using the bare `kxvo` or `python` commands below.

```sh
uv venv --python 3.11
uv pip install -e '.[test,xfeat,plots,benchmark,sitl]'
.venv/bin/python -m pytest -q
scripts/setup-xfeat.sh
KXVO_PYTHON=.venv/bin/python scripts/build-sitl.sh
```

Start with a synthetic smoke test. It checks the code path from the official checkpoint through training and validation to a reloadable model.

```sh
.venv/bin/python scripts/make-smoke-data.py runs/my-smoke
OMP_NUM_THREADS=2 .venv/bin/python -m kxvo.train \
  --manifest runs/my-smoke/manifest.csv --output runs/my-training \
  --checkpoint runs/xfeat-upstream/weights/xfeat.pt \
  --steps 3 --validate-every 1 --width 160 --height 128 --batch 2
```

For the actual experiment, supply a CSV with `path,split,group,rights` and add camera/calibration and capture provenance columns. Paths are relative to the manifest. Assign whole flight sessions to train, validation and test. Pick dimensions divisible by 32 that preserve the capture aspect ratio. Start with two seeds, use `--device cuda` on a rented CUDA machine if needed, and save the validation logs and selected checkpoint for each seed. A seed can legitimately have no selected checkpoint if it fails the clean-recall or blur-improvement check.

The default training recipe updates the descriptor backbone with known warp correspondences. Add `--train-detector` to train the keypoint head using clean detections from the pinned XFeat teacher projected into the warped view. Training includes clean pairs and blur on one or both images. Keep descriptor-only and detector-training results labelled separately.

### Replay and score

New live recordings store lossless full-resolution color arrays by default. Use `--record-format png` for a short capture that needs portable image files. Raw 720p color capture at 60 fps uses about 10 GB per minute. Replay converts and resizes them to the requested calibrated processing width. Old recordings can still replay, with the image detail available in the saved frames. Use the same camera, mount offsets and recording for all trackers. An XFeat run requires a local `--checkpoint`. ORB and XFeat default to hybrid tracking. Their detections seed tracks, KLT tracks them at subpixel resolution, and descriptors re-associate lost landmarks. Use `--association descriptor` to measure direct descriptor matching with geometric filtering. Use `--landmark-gap 0` and an explicit `--output` for the no-recovery ablation. Recordings and replays refuse to overwrite existing outputs.

```sh
kxvo replay --camera runs/my-smoke/camera.json --range-offset 0 0 0 runs/my-smoke/blur
kxvo replay --camera runs/my-smoke/camera.json --range-offset 0 0 0 \
  --tracker orb runs/my-smoke/blur
kxvo replay --camera runs/my-smoke/camera.json --range-offset 0 0 0 \
  --tracker xfeat --checkpoint runs/xfeat-upstream/weights/xfeat.pt runs/my-smoke/blur
kxvo replay --camera runs/my-smoke/camera.json --range-offset 0 0 0 \
  --tracker xfeat --checkpoint runs/my-training/seed-1/step-3.pt runs/my-smoke/blur
python -m kxvo.score --reference runs/my-smoke/blur/log.jsonl \
  --output results/my-replay.json --plot results/my-replay.png runs/my-smoke/blur/replay-*.jsonl
```

For repeated real replay, `scripts/replay-experiment.sh CAMERA_JSON RUN_DIR OUTPUT_DIR CHECKPOINT [CHECKPOINT ...]` runs each tracker once with a declared seed using the default FlyTimes mount offsets. Set `KXVO_WIDTH` if your capture profile differs.

For real flights, `python -m kxvo.ground_truth --help` describes how to produce a reference CSV from a surveyed overhead camera and a top-mounted ArUco marker. Supply one synchronized timestamp per video frame. Use one clean alignment window shared by all arms. Report the reference error and any periods without marker coverage. Start with a clean development clip to verify geometry, scale, timing and attitude/range before judging the learned tracker.

Run the same sparse MegaDepth-1500 protocol with each checkpoint and seed. Download the dataset using the [pinned upstream instructions](https://github.com/verlab/accelerated_features#evaluation). The dataset flag points at the directory containing `megadepth_test_1500`.

```sh
python -m kxvo.megadepth --dataset runs/datasets/Mega1500 \
  --checkpoint runs/xfeat-upstream/weights/xfeat.pt --seed 1
```

### ArduPilot SITL

The runner starts a local Copter 4.7.1 child, renders a textured room at the live simulated aircraft pose and reads attitude and range from that autopilot over MAVLink. It establishes the origin and takes off with simulated GPS available, then switches position and yaw to ExternalNav. This checks GPS-to-ExternalNav transition. It exercises force-arm only inside the runner's private SITL process. Use a different `--instance` for simultaneous runs.

```sh
python -m kxvo.sitl --arducopter runs/ardupilot/build/sitl/bin/arducopter \
  --output runs/my-truth --fixture
python -m kxvo.sitl --arducopter runs/ardupilot/build/sitl/bin/arducopter \
  --output runs/my-stock
OMP_NUM_THREADS=2 python -m kxvo.sitl --arducopter runs/ardupilot/build/sitl/bin/arducopter \
  --output runs/my-xfeat --tracker xfeat --checkpoint runs/xfeat-upstream/weights/xfeat.pt
```

Use the fine-tuned file in the same command to test that model. To exercise assumed link degradation, add `--delay-ms 80 --jitter-ms 15 --drop 0.1 --exposure-ms 12 --blur-min-px 7 --blackout-seconds 1 --downlink-bps 1250 --fallback-source flow`. This uses an explicit synthetic blur floor and simulates a pilot switch from ExternalNav to optical flow. GPS supplies takeoff and the initial position in this scenario. The effective parameter file records the delay compensation. The summary checks the pose stream, actual hold drift, estimator error and valid coverage, in addition to EKF evidence and landing. Replace the injected assumptions with FlyTimes measurements before using the simulation to predict hardware behaviour.

## Limits

- The floor must be flat, textured and well lit (the MTF-02P also needs at least 60 lux). Plain glossy floors and dark halls will fail.
- The rangefinder must see the same floor the camera sees. Flying over tables or steps corrupts scale.
- There is no loop closure and no relocalisation. A `lost` status restarts the map and increments `reset_counter`.
- On Betaflight, P3 has no latency compensation.
- Indoor flight only, in a fully enclosed hall with venue permission, with a pilot ready to take over and the source switch within reach. Outdoor automatic modes need the PDRA01.

## Layout

| Path | Contents |
|---|---|
| `kxvo/vo.py` | VO core: tracking, landmarks, pose |
| `kxvo/pipeline.py` | Camera to body NED, height from slant range |
| `kxvo/synth.py`, `kxvo/bench.py` | Synthetic room and benchmark |
| `kxvo/mav.py` | MAVLink telemetry in, pose out |
| `kxvo/video.py`, `kxvo/calib.py` | Capture, recording, replay, calibration |
| `params/` | ArduPilot and KX Betaflight starting parameters |

Licence: Apache-2.0.
