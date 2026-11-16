"""Run a versioned odometry replay scenario and save its scorecard."""

import argparse
import json
import math
from pathlib import Path

from .cli import main as vo
from .score import load_reference, score
from .trackers import sha256


def run_pose_stream(manifest, config, output, arducopter, instance):
    from .sitl import main as sitl

    output.mkdir(parents=True)
    flight = output / "flight"
    result = {"scenario": config.get("name", manifest.stem), "track": "pose_stream",
              "manifest_sha256": sha256(manifest)}
    try:
        argv = ["--arducopter", str(arducopter), "--output", str(flight),
                "--seed", str(config["seed"]), "--instance", str(instance), "--timeout", "110"]
        if config.get("pose_stream"):
            argv += ["--pose-stream", str(manifest.parent / config["pose_stream"])]
        else:
            argv += ["--fixture"]
        fault = config.get("fault", "none")
        if fault == "outlier":
            offset = float(config["offset_m"])
            if not 0 < offset <= 10:
                raise ValueError("Outlier must be in (0, 10] metres")
            argv += ["--outlier-m", str(offset), "--full-covariance"]
        elif fault == "stale":
            argv += ["--stop-poses-after", str(config["after_s"])]
        elif fault != "none":
            raise ValueError(f"Unsupported pose-stream fault: {fault}")
        status = sitl(argv)
        summary = json.loads((flight / "summary.json").read_text())
        if not summary["dataflash"]:
            raise ValueError("Missing DataFlash evidence from the simulator")
        result.update(verdict="pass" if status == 0 else "fail", evidence=summary)
    except (KeyError, TypeError, ValueError, OSError, SystemExit) as exc:
        result.update(verdict="invalid", reason=str(exc))
    (output / "score.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{result['verdict']}: {output / 'score.json'}")
    return {"pass": 0, "fail": 1, "invalid": 2}[result["verdict"]]


def run(manifest, output, tracker, checkpoint, device, window_details=None):
    config = json.loads(manifest.read_text())
    if (not isinstance(config, dict) or config.get("version") != 1
            or config.get("track") != "odometry_replay"):
        raise ValueError("Expected a version 1 odometry_replay scenario")
    output.mkdir(parents=True)
    trace = output / "poses.jsonl"
    result = {"scenario": config.get("name", manifest.stem), "manifest_sha256": sha256(manifest),
              "track": config["track"], "tracker": tracker, "device": device}
    try:
        camera = manifest.parent / config["camera"]
        recording = manifest.parent / config["recording"]
        reference = manifest.parent / config["reference"]
        result.update(camera_sha256=sha256(camera), reference_sha256=sha256(reference),
                      recording_log_sha256=sha256(recording / "log.jsonl"),
                      checkpoint_sha256=sha256(checkpoint) if checkpoint else None)
        argv = ["replay", str(recording), "--camera", str(camera), "--width", str(config["width"]),
                "--seed", str(config["seed"]), "--tracker", tracker, "--device", device,
                "--range-offset", "0", "0", str(config["range_offset_z_m"]), "--output", str(trace)]
        if checkpoint:
            argv += ["--checkpoint", str(checkpoint)]
        vo(argv)
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        metrics, _ = score(rows, load_reference(reference), alignment_start=config["alignment_start_s"],
                           window_details=window_details)
        limits = config["limits"]
        if not (0 <= limits["min_valid_fraction"] <= 1 and
                math.isfinite(limits["max_relative_translation_m"]) and limits["max_relative_translation_m"] > 0):
            raise ValueError("Invalid scenario limits")
        if not metrics["reference_covered_frames"]:
            raise ValueError("No timestamps covered by the reference")
        failures = []
        if metrics["valid_fraction"] < limits["min_valid_fraction"]:
            failures.append("valid fraction below limit")
        relative = metrics.get("relative_translation_m_median")
        if relative is None:
            failures.append("no continuous 5 s relative-error samples")
        elif relative > limits["max_relative_translation_m"]:
            failures.append("5 s relative translation error exceeds limit")
        result.update(verdict="fail" if failures else "pass", failures=failures, metrics=metrics, limits=limits)
    except (KeyError, TypeError, ValueError, OSError, SystemExit) as exc:
        result.update(verdict="invalid", reason=str(exc))
    (output / "score.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{result['verdict']}: {output / 'score.json'}")
    return {"pass": 0, "fail": 1, "invalid": 2}[result["verdict"]]


def make_fixture(root, scene_seed=12, sensor_seed=8, degraded=False):
    from .geometry import rot_to_euler
    from .train import motion_blur
    from .synth import Room, default_camera, simulate
    from .video import Recorder

    root.mkdir(parents=True)
    camera = default_camera().scaled(320)
    camera.save(root / "camera.json")
    recording = Recorder(root / "capture")
    try:
        for sample in simulate(duration=12, fps=20, cam=camera, room=Room(seed=scene_seed), seed=sensor_seed):
            image = sample.image
            if degraded and 3.5 <= sample.t % 6 < 4.5:
                image = motion_blur(image, 7, 0)
            recording.write(sample.t, image, {
                "att": sample.att_meas, "range_m": sample.range_m, "truth_p": sample.p_WB.tolist(),
                "truth_rpy": rot_to_euler(sample.R_WB),
            })
    finally:
        recording.close()
    manifest = root / "scenario.json"
    manifest.write_text(json.dumps({
        "version": 1, "name": "synthetic-room-replay", "track": "odometry_replay",
        "camera": "camera.json", "recording": "capture", "reference": "capture/log.jsonl",
        "width": 320, "seed": 1, "range_offset_z_m": 0, "alignment_start_s": 1,
        "limits": {"min_valid_fraction": 0.9, "max_relative_translation_m": 0.4}
    }, indent=2) + "\n")
    return manifest


def quickstart(root):
    return run(make_fixture(root), root / "result", "klt", None, "cpu")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    demo = commands.add_parser("quickstart", help="generate and score a redistributable synthetic recording")
    demo.add_argument("output", type=Path)
    replay = commands.add_parser("run", help="run an odometry replay scenario")
    replay.add_argument("scenario", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--tracker", choices=["klt", "orb", "xfeat"], default="klt")
    replay.add_argument("--checkpoint", type=Path)
    replay.add_argument("--device", default="cpu")
    replay.add_argument("--arducopter", type=Path, default=Path("runs/ardupilot/build/sitl/bin/arducopter"))
    replay.add_argument("--instance", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.action == "quickstart":
            return quickstart(args.output)
        config = json.loads(args.scenario.read_text())
        if isinstance(config, dict) and config.get("version") == 1 and config.get("track") == "pose_stream":
            return run_pose_stream(args.scenario, config, args.output, args.arducopter, args.instance)
        return run(args.scenario, args.output, args.tracker, args.checkpoint, args.device)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
