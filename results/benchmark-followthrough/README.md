# Geometric XFeat and ArduPilot benchmark results

## Geometric supervision

Two seeds per arm started from the same pretrained XFeat checkpoint and trained for 1,000 steps on 180 anchors. Sixty same-camera pairs on separate routes supplied validation. Both arms used the same validation pairs. The geometric arm used actual frames two ticks apart, with calibrated depth, NED pose and forward/backward occlusion checks. The control used homographies on the same anchor images. The detector uses clean detections from each actual view in the geometric arm and projected clean detections in the warp control.

| Weights | Validation clean recall | Validation blur recall |
| --- | ---: | ---: |
| Pretrained XFeat | 30.1% | 21.9% |
| Geometric seed 1 | 28.7% | 22.0% |
| Geometric seed 2 | 28.5% | 21.9% |
| Warp seed 1 | 26.7% | 20.8% |
| Warp seed 2 | 26.6% | 20.7% |

Geometric supervision preserved more viewpoint matching than the warp control. Neither geometric seed reached the minimum one-point improvement in blurred recall against pretrained XFeat. The comparison used the development split. The held-out school and MegaDepth remain reserved for a qualifying run. [Metrics and hashes](geometric-training.json) · [TartanAir camera and depth formats](https://tartanair.org/modalities.html)

## ArduPilot

| Native fallback profile | Passes | Hold drift, median and range | Hold XY estimator RMS, median and range |
| --- | ---: | ---: | ---: |
| Stock KLT | 3/3 | 0.042 m (0.035–0.043) | 0.044 m (0.035–0.045) |
| Pretrained XFeat | 3/3 | 0.030 m (0.022–0.035) | 0.042 m (0.036–0.063) |

The fallback profile injects video latency, drops, blur and a blackout. DataFlash confirms the change to optical flow in all six runs. A separate continuous ExternalNav run with stock KLT reached all four 0.7 m box waypoints, landed and disarmed, and recorded 334 box frames with at least 3 px of motion-derived blur. The 55 ms synthetic exposure produced a 9 px peak during turns. [Native repeats](native-fallback.json) · [Box](native-box.json)

The pose-stream track passed the 5 m outlier case and failed the deliberate stale-pose case. Its inputs come straight from simulator truth, independent of RGB estimates. The stationary ArduPilot JSON clock fixture completed 2,000 ticks in each of three normal runs and three runs with two busy CPU workers. Withholding updates for 0.5 s caused no counter advance. All six traces had the same hash. This exercises protocol timing with stationary sensor inputs. [Pose-stream verdicts](pose-stream.json) · [JSON clock](json-clock.json) · [ArduPilot JSON documentation](https://ardupilot.org/dev/docs/sitl-with-JSON.html)

## Gazebo

Gazebo Harmonic passed a north-aligned clean flight and a scripted GPS recovery with the synthetic renderer. The mounted Gazebo RGB camera also passed clean ExternalNav and blackout-to-GPS recovery with a 15° downward mount and a textured floor. The clean camera run measured 0.093 m horizontal estimator RMS and 0.036 m hold drift. DataFlash confirms the source change in the recovery run.

Earlier RGB variants are preserved in the evidence. The first lost tracking. The flight-envelope guard led to a controlled landing in the next run. Correcting the camera inertia restored valid pose coverage, and the upward-facing camera still exceeded the estimator error limit. The downward-facing profile passed the existing limits. Its range input comes from an emulated plane, and it uses host receipt times to associate Gazebo RGB with simulator truth. Tick-synchronized camera scoring and physical flight remain open. [Gazebo evidence](gazebo-physics.json) · [ArduPilot Gazebo documentation](https://ardupilot.org/dev/docs/sitl-with-gazebo.html)
