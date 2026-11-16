"""Run a versioned odometry replay scenario and save its scorecard."""

import argparse
import json
import math
from pathlib import Path

from .cli import main as vo
from .score import load_reference, score
from .trackers import sha256


def run(manifest, output, tracker, checkpoint, device):
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
        metrics, _ = score(rows, load_reference(reference), alignment_start=config["alignment_start_s"])
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


def quickstart(root):
    from .geometry import rot_to_euler
    from .synth import Room, default_camera, simulate
    from .video import Recorder

    root.mkdir(parents=True)
    camera = default_camera().scaled(320)
    camera.save(root / "camera.json")
    recording = Recorder(root / "capture")
    try:
        for sample in simulate(duration=12, fps=20, cam=camera, room=Room(seed=12), seed=8):
            recording.write(sample.t, sample.image, {
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
    return run(manifest, root / "result", "klt", None, "cpu")


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
    args = parser.parse_args(argv)
    try:
        if args.action == "quickstart":
            return quickstart(args.output)
        return run(args.scenario, args.output, args.tracker, args.checkpoint, args.device)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
