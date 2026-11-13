# kxvo

Camera-only monocular visual odometry for small indoor drones, running on a ground computer. It reads the FPV video feed and the flight controller's rangefinder and attitude, and publishes the drone's position as MAVLink `VISION_POSITION_ESTIMATE` back to the flight controller.

Built for the Kytona FlyTimes 85 HDZero test drone (GOKU F405 HD, MTF-02P flow and range sensor). ArduPilot and the KX Betaflight fork (patch P3) both accept its output.

**Status: tested on synthetic video only. Not yet run on real footage, in SITL against a flight controller, or in flight.**

## How it works

Monocular VO cannot measure scale. kxvo takes scale from the floor:

1. The downward rangefinder gives camera height. Corners that look clearly below the horizon are placed on the floor plane in metres. A corner is accepted as floor only if the floor point also lines up with where the corner was first seen, from a camera position at least 8 cm away. That two-view check keeps wall corners from being treated as floor.
2. Corners on walls and furniture are triangulated once the drone has moved enough.
3. Each frame's pose is solved from this small map. Floor landmarks lead whenever there are at least 25 of them, so scale stays tied to the rangefinder. Pose solving is a guided, trimmed PnP from a constant-velocity prediction, refined with camera height held to the rangefinder reading.
4. Tracks are re-tracked against a keyframe every 8 frames to limit tracking drift.

The flight controller's attitude is used only to level the start pose and to turn the slant range into vertical height. This is loose use of the FC, not tightly coupled visual-inertial odometry.

kxvo publishes estimates only. It never sends arming, mode or setpoint commands. When it is not confident it stops publishing, and the FC falls back to its own sources on its own timeout.

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
.venv/bin/python -m pytest -q          # 13 tests
.venv/bin/kxvo synth --seeds 5         # the benchmark above
```

## Hardware path

```
FlyTimes camera -> HDZero VTX -> BoxPro goggles -> mini HDMI out -> USB capture dongle -> Mac (kxvo)
Mac (kxvo) -> data link -> GOKU F405 UART -> ArduPilot VISO / KX Betaflight P3
GOKU F405 -> data link -> Mac: ATTITUDE, DISTANCE_SENSOR (MTF-02P)
```

**Data link without a Wi-Fi board.** kxvo sends compact messages, about 45 bytes each, at up to 10 Hz (`--send-hz`): about 0.5 kB/s, plus attitude and range telemetry coming back.

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
