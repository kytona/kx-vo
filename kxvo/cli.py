"""kxvo command line: synthetic benchmark, live run, replay and calibration."""

from __future__ import annotations

import argparse
import json
import math
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
    import cv2

    from .camera import Camera
    from .pipeline import Mount, Pipeline
    from .vo import VOConfig

    cv2.setRNGSeed(a.seed)

    cam = Camera.load(a.camera)
    if cam.width != a.width:
        cam = cam.scaled(a.width)
    mount = Mount(camera_B=tuple(a.camera_offset), rangefinder_B=tuple(a.range_offset))
    tracker = None
    if a.checkpoint and a.tracker != "xfeat":
        raise SystemExit("--checkpoint requires --tracker xfeat")
    if not 0 <= a.landmark_gap <= 2:
        raise SystemExit("--landmark-gap must be between 0 and 2 seconds")
    if a.tracker != "klt":
        from .trackers import DescriptorTracker
        try:
            tracker = DescriptorTracker(a.tracker, 300, a.checkpoint, a.xfeat_source, a.device, a.landmark_gap, a.association)
        except (ValueError, RuntimeError, OSError) as exc:
            raise SystemExit(str(exc)) from exc
    if a.reproj_px is not None and not (math.isfinite(a.reproj_px) and a.reproj_px > 0):
        raise SystemExit("--reproj-px must be finite and positive")
    config = VOConfig()
    if tracker is not None:
        px_scale = cam.width / 320
        config.pnp_reproj_px = (a.reproj_px or 3.0) * px_scale
        config.cull_reproj_px = config.pnp_reproj_px * 1.5
        config.verify_reproj_px = config.pnp_reproj_px * 0.75
    elif a.reproj_px is not None:
        config.pnp_reproj_px = a.reproj_px * cam.width / 320
    return cam, Pipeline(cam, mount, config, tracker=tracker)


def _record(out, rng, att):
    p = out.p_WB
    return {
        "status": out.status, "valid": out.valid,
        "p": [round(float(v), 4) for v in p], "rpy": [round(float(v), 5) for v in out.rpy],
        "tracks": out.vo.n_tracks, "landmarks": out.vo.n_landmarks, "inliers": out.vo.n_inliers,
        "reset": out.reset_counter, "sigma": round(out.pos_sigma_m, 3),
        "range_m": rng, "att": list(att) if att else None, "tracking": out.vo.info,
    }


def can_publish_pose(valid, distance, attitude, capture_t, now, max_age):
    return (valid and distance is not None and math.isfinite(distance) and distance > 0
            and attitude is not None and all(map(math.isfinite, attitude))
            and 0 <= now - capture_t <= max_age)


def cmd_run(a):
    from .video import Recorder, VideoSource

    if a.send and (not a.mavlink or not a.video.isdigit()):
        raise SystemExit("--send requires live capture and --mavlink")
    if not 0 < a.max_frame_age_ms <= 250 or not 0 < a.send_hz <= 10:
        raise SystemExit("Use send rate in (0, 10] Hz and frame age in (0, 250] ms")
    cam, pipe = _pipeline(a)
    video = VideoSource(a.video, cam.width, a.video_latency_ms / 1000.0)
    mav = None
    if a.mavlink:
        from .mav import MavLink
        mav = MavLink(a.mavlink)
    rec = Recorder(a.record, frame_format=a.record_format) if a.record else None
    meta = _run_meta(a, pipe)
    if rec:
        (rec.dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("kxvo running. Ctrl-C to stop." + ("" if a.send else " Not sending to the FC (add --send)."))
    n, t0, sent = 0, time.monotonic(), 0
    next_send = 0.0
    period = 1.0 / a.send_hz
    try:
        while True:
            t, gray, frame = video.read()
            if gray is None:
                break
            if gray.shape != (cam.height, cam.width):
                raise ValueError(f"Capture aspect ratio differs from calibration: {frame.shape[:2]}")
            att, rng, ages = mav.snapshot_with_ages() if mav else (None, None, {})
            if mav:
                mav.heartbeat()
            start = time.monotonic()
            out = pipe.step(gray, t, rng, att)
            processing_ms = (time.monotonic() - start) * 1000
            send_time = None
            now = time.monotonic()
            if a.send and now >= next_send and can_publish_pose(out.valid, rng, att, t, now, a.max_frame_age_ms / 1000):
                mav.send_pose(t, out.p_WB, out.rpy, out.pos_sigma_m, out.reset_counter,
                              compact=not a.full_covariance)
                send_time = time.monotonic()
                sent += 1
                next_send = now + period
            if rec:
                rec.write(t, frame, {**_record(out, rng, att), "processing_ms": processing_ms,
                                    "capture_size": [frame.shape[1], frame.shape[0]],
                                    "pose_sent_t": send_time, "telemetry_age_s": ages})
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


def _run_meta(a, pipe):
    tracker = pipe.vo.tracker
    return {"tracker": a.tracker, "checkpoint_sha256": tracker.checkpoint_hash if tracker else None,
            "association": a.association if tracker else "klt", "pnp_reproj_px": pipe.vo.cfg.pnp_reproj_px,
            "landmark_gap_s": a.landmark_gap, "opencv_seed": a.seed, "width": pipe.cam.width}


def cmd_replay(a):
    from .video import replay_records

    cam, pipe = _pipeline(a)
    meta = _run_meta(a, pipe)
    name = meta["tracker"]
    if meta["checkpoint_sha256"]:
        name += "-" + meta["checkpoint_sha256"][:12]
    out_path = Path(a.output) if a.output else Path(a.run) / f"replay-{name}-seed{a.seed}.jsonl"
    n = 0
    with out_path.open("x") as f:
        for t, gray, rng, att in replay_records(a.run, cam.width):
            if gray.shape != (cam.height, cam.width):
                raise ValueError("Recording aspect ratio differs from calibration")
            start = time.monotonic()
            out = pipe.step(gray, t, rng, att)
            f.write(json.dumps({"t": t, **_record(out, rng, att), **meta,
                                "processing_ms": (time.monotonic() - start) * 1000}) + "\n")
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
        p.add_argument("--tracker", choices=("klt", "orb", "xfeat"), default="klt")
        p.add_argument("--checkpoint", type=Path)
        p.add_argument("--association", choices=("hybrid", "descriptor"), default="hybrid")
        p.add_argument("--xfeat-source", type=Path, default=Path("runs/xfeat-upstream"))
        p.add_argument("--device", default="cpu")
        p.add_argument("--landmark-gap", type=float, default=0.5)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--reproj-px", type=float, help="PnP threshold in pixels at equivalent 320 px width")
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
    r.add_argument("--max-frame-age-ms", type=float, default=200.0)
    r.add_argument("--send", action="store_true", help="publish VISION_POSITION_ESTIMATE to the FC")
    r.add_argument("--send-hz", type=float, default=10.0,
                   help="max publish rate; 10 Hz compact is about 0.5 kB/s, sized for ELRS links")
    r.add_argument("--full-covariance", action="store_true",
                   help="send VO covariance (about 1.3 kB/s at 10 Hz; Wi-Fi or USB links only)")
    r.add_argument("--record-format", choices=("npy", "png"), default="npy",
                   help="Lossless source frame format for live recordings")
    r.add_argument("--record", help="directory to save frames and log for replay")
    r.add_argument("--show", action="store_true")
    r.set_defaults(fn=cmd_run)

    p = sub.add_parser("replay", help="rerun VO on a recorded run directory")
    common(p)
    p.add_argument("run")
    p.add_argument("--output")
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
