"""kxvo command line: synthetic benchmark, live run, replay and calibration."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def cmd_synth(a):
    from .bench import run_synth

    results = []
    for seed in range(1, a.seeds + 1):
        r = run_synth(duration=a.duration, seed=seed)
        results.append(r)
        print(f"seed {seed}: {r.summary()}")
    rmse = np.mean([r.rmse_m for r in results])
    mx = np.mean([r.max_err_m for r in results])
    path = np.mean([r.path_m for r in results])
    print(f"mean over {len(results)} seeds: horizontal RMSE {rmse:.3f} m, max {mx:.3f} m, "
          f"path {path:.1f} m ({100 * rmse / path:.1f}% RMSE/path). Synthetic best case, not flight evidence.")


def _pipeline(a):
    from .camera import Camera
    from .pipeline import Mount, Pipeline

    cam = Camera.load(a.camera)
    if cam.width != a.width:
        cam = cam.scaled(a.width)
    mount = Mount(camera_B=tuple(a.camera_offset), rangefinder_B=tuple(a.range_offset))
    return cam, Pipeline(cam, mount)


def _record(out, rng, att):
    p = out.p_WB
    return {
        "status": out.status, "valid": out.valid,
        "p": [round(float(v), 4) for v in p], "rpy": [round(float(v), 5) for v in out.rpy],
        "tracks": out.vo.n_tracks, "landmarks": out.vo.n_landmarks, "inliers": out.vo.n_inliers,
        "reset": out.reset_counter, "sigma": round(out.pos_sigma_m, 3),
        "range_m": rng, "att": list(att) if att else None,
    }


def cmd_run(a):
    from .video import Recorder, VideoSource

    cam, pipe = _pipeline(a)
    video = VideoSource(a.video, cam.width, a.video_latency_ms / 1000.0)
    mav = None
    if a.mavlink:
        from .mav import MavLink
        mav = MavLink(a.mavlink)
    rec = Recorder(a.record) if a.record else None
    if a.send and mav is None:
        sys.exit("--send needs --mavlink")
    print("kxvo running. Ctrl-C to stop." + ("" if a.send else " Not sending to the FC (add --send)."))
    n, t0, sent = 0, time.monotonic(), 0
    next_send = 0.0
    period = 1.0 / a.send_hz
    try:
        while True:
            t, gray, frame = video.read()
            if gray is None:
                break
            att, rng = mav.snapshot() if mav else (None, None)
            if mav:
                mav.heartbeat()
            out = pipe.step(gray, t, rng, att)
            now = time.monotonic()
            if a.send and out.valid and rng is not None and now >= next_send:
                mav.send_pose(t, out.p_WB, out.rpy, out.pos_sigma_m, out.reset_counter,
                              compact=not a.full_covariance)
                sent += 1
                next_send = now + period
            if rec:
                rec.write(t, gray, _record(out, rng, att))
            n += 1
            if a.show:
                import cv2
                vis = frame.copy()
                s = vis.shape[1] / cam.width
                for x, y in pipe.vo.pts:
                    cv2.circle(vis, (int(x * s), int(y * s)), 2, (0, 255, 0), -1)
                cv2.putText(vis, f"{out.status} p={np.round(out.p_WB, 2)} yaw={math.degrees(out.rpy[2]):.1f}",
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow("kxvo", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if n % 30 == 0:
                fps = n / (time.monotonic() - t0)
                print(f"{fps:5.1f} fps {out.status:8s} p={np.round(out.p_WB, 2)} inliers={out.vo.n_inliers} "
                      f"range={rng} sent={sent}")
    except KeyboardInterrupt:
        pass
    finally:
        video.close()
        if rec:
            rec.close()
        if mav:
            mav.close()


def cmd_replay(a):
    from .video import replay_records

    cam, pipe = _pipeline(a)
    out_path = Path(a.run) / "kxvo_replay.jsonl"
    n = 0
    with out_path.open("w") as f:
        for t, gray, rng, att in replay_records(a.run):
            out = pipe.step(gray, t, rng, att)
            f.write(json.dumps({"t": t, **_record(out, rng, att)}) + "\n")
            n += 1
    print(f"replayed {n} frames -> {out_path}")


def cmd_calibrate(a):
    from .calib import calibrate

    cols, rows = (int(v) for v in a.board.lower().split("x"))
    cam = calibrate(a.video, cols, rows, a.square, a.width, a.fisheye, a.uptilt, want=a.views, show=not a.no_show)
    cam.save(a.out)
    print(f"saved {a.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kxvo", description="Camera-only visual odometry for small drones")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="benchmark on the synthetic room")
    s.add_argument("--duration", type=float, default=24.0)
    s.add_argument("--seeds", type=int, default=3)
    s.set_defaults(fn=cmd_synth)

    def common(p):
        p.add_argument("--camera", required=True, help="camera JSON from kxvo calibrate")
        p.add_argument("--width", type=int, default=320, help="processing width in pixels")
        p.add_argument("--camera-offset", type=float, nargs=3, default=[0.03, 0.0, 0.0], metavar=("X", "Y", "Z"),
                       help="camera position in body FRD, m")
        p.add_argument("--range-offset", type=float, nargs=3, default=[0.0, 0.0, 0.01], metavar=("X", "Y", "Z"),
                       help="rangefinder position in body FRD, m")

    r = sub.add_parser("run", help="live VO from a capture device or file")
    common(r)
    r.add_argument("--video", default="0", help="capture index (HDMI dongle) or file path")
    r.add_argument("--video-latency-ms", type=float, default=0.0)
    r.add_argument("--mavlink", help="e.g. udpin:0.0.0.0:14550 or /dev/cu.usbmodem1:460800")
    r.add_argument("--send", action="store_true", help="publish VISION_POSITION_ESTIMATE to the FC")
    r.add_argument("--send-hz", type=float, default=10.0,
                   help="max publish rate; 10 Hz compact is about 0.5 kB/s, sized for ELRS links")
    r.add_argument("--full-covariance", action="store_true",
                   help="send VO covariance (about 1.3 kB/s at 10 Hz; Wi-Fi or USB links only)")
    r.add_argument("--record", help="directory to save frames and log for replay")
    r.add_argument("--show", action="store_true")
    r.set_defaults(fn=cmd_run)

    p = sub.add_parser("replay", help="rerun VO on a recorded run directory")
    common(p)
    p.add_argument("run")
    p.set_defaults(fn=cmd_replay)

    c = sub.add_parser("calibrate", help="checkerboard calibration through the real video path")
    c.add_argument("--video", default="0")
    c.add_argument("--board", default="9x6", help="inner corners, cols x rows")
    c.add_argument("--square", type=float, required=True, help="square size in metres")
    c.add_argument("--width", type=int, default=320)
    c.add_argument("--fisheye", action="store_true")
    c.add_argument("--uptilt", type=float, default=20.0, help="camera uptilt on the airframe, degrees")
    c.add_argument("--views", type=int, default=25)
    c.add_argument("--no-show", action="store_true")
    c.add_argument("--out", default="camera.json")
    c.set_defaults(fn=cmd_calibrate)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
