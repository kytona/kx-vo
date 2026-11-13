"""Run kx-vo against a private ArduCopter SITL process and render its camera."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import signal
import socket
import subprocess
import time

import numpy as np

from .geometry import camera_to_body, euler_to_rot
from .mav import mavutil, vision_position_estimate
from .pipeline import Mount, Pipeline
from .synth import Room, default_camera, render
from .trackers import DescriptorTracker, sha256
from .train import motion_blur
from .video import Recorder


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def command(mav, command_id, *params):
    values = list(params) + [0] * (7 - len(params))
    mav.mav.command_long_send(1, 1, command_id, 0, *values)


def request_message(mav, msg_id, hz):
    command(mav, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 1e6 / hz)


def local_position(m, origin):
    lat, lon, alt = origin
    radius = 6378137
    return np.array([math.radians(m.lat_int * 1e-7 - lat) * radius,
                     math.radians(m.lon_int * 1e-7 - lon) * radius * math.cos(math.radians(lat)),
                     alt - m.alt])


def ekf_evidence(log, outlier_usec=None, outlier_m=0):
    mav = mavutil.mavlink_connection(str(log))
    counts = {"secondary_samples": 0, "vision_received": 0, "vision_accepted": 0,
              "secondary_horizontal_innovation_samples": 0, "land_mode": False}
    source = {}
    outlier_time = None
    outlier_innovations = {}
    outlier_ratio = None
    try:
        while (msg := mav.recv_match(type=["XKFS", "VISP", "XKF3", "XKF4", "MODE"])) is not None:
            kind = msg.get_type()
            if kind == "XKFS":
                source[msg.C] = msg.SS
                counts["secondary_samples"] += int(msg.SS == 1)
            elif kind == "VISP":
                counts["vision_received"] += 1
                counts["vision_accepted"] += int(msg.Ign == 0)
                if outlier_usec is not None and msg.RTimeUS == outlier_usec:
                    outlier_time = msg.TimeUS
            elif kind in {"XKF3", "XKF4"} and source.get(msg.C) == 1:
                if outlier_time is not None and 0 <= msg.TimeUS - outlier_time < 600000:
                    key = msg.TimeUS, msg.C
                    sample = outlier_innovations.setdefault(key, {})
                    sample[kind] = msg.IPN if kind == "XKF3" else msg.SP
                    if (sample.get("XKF4", 0) > 1
                            and abs(sample.get("XKF3", 0) + outlier_m) <= abs(outlier_m) * 0.35):
                        outlier_ratio = max(outlier_ratio or 0, sample["XKF4"])
                if kind != "XKF3":
                    continue
                counts["secondary_horizontal_innovation_samples"] += int(
                    msg.IPN != 0 or msg.IPE != 0)
            elif kind == "MODE":
                counts["land_mode"] |= msg.Mode == 9
    finally:
        mav.close()
    if outlier_usec is not None:
        counts["outlier_received"] = outlier_time is not None
        counts["outlier_position_gate_exceeded"] = outlier_ratio is not None
        counts["outlier_position_test_ratio_sqrt"] = outlier_ratio
    return counts


def run(args, mav, recorder):
    rng = np.random.default_rng(args.seed)
    camera = default_camera()
    camera.save(args.output / "camera.json")
    mount = Mount(rangefinder_B=(0, 0, 0))
    tracker = None if args.tracker == "klt" else DescriptorTracker(
        args.tracker, 300, args.checkpoint, args.xfeat_source)
    pipe = Pipeline(camera, mount, tracker=tracker)
    room = Room(seed=args.seed)
    queue = deque(maxlen=5)
    epoch = time.monotonic()
    next_send, last_capture, last_processed, last_pose_send = 0, -1, -1, -1
    last_heartbeat = last_control = -1
    phase = "startup"
    sent = frames = dropped = valid = 0
    origin = start_pos = None
    armed_at = switched_at = None
    first_truth = None
    states = {}
    telemetry = {}
    downlink_next = 0.0
    downlink_dropped = 0
    statuses = []
    hold = []
    hold_truth = []
    hold_poses = 0
    last_hold_pose = None
    hold_pose_gaps = []
    blur_lengths = []
    outlier_target = None
    outlier_usec = None
    landing_at = None
    pose_bytes = 0
    injected_outlier = False
    fallback_sent = False
    peak_queue = 0
    armed = False
    received_since = None
    stream_setup = False
    status_file = args.output / "flight.jsonl"
    interval = max(0.1, (129 if args.full_covariance else 44) / args.uplink_bps)
    if interval > 0.25:
        raise ValueError("Uplink budget cannot carry four nominal poses per second")

    with status_file.open("x") as trace:
        while (now := time.monotonic() - epoch) < args.timeout:
            message = mav.recv_match(blocking=True, timeout=0.01)
            if now - last_heartbeat >= 1:
                mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, 0, 0, 0, 0)
                last_heartbeat = now
            if message is not None:
                kind = message.get_type()
                states[kind] = (now, message)
                if kind in {"ATTITUDE", "DISTANCE_SENSOR"}:
                    earliest = max(now, downlink_next)
                    available = earliest + (len(message.get_msgbuf()) / args.downlink_bps if args.downlink_bps else 0)
                    if available - now <= 0.2:
                        downlink_next = available
                        telemetry.setdefault(kind, deque(maxlen=16)).append((available, now, message))
                    else:
                        downlink_dropped += 1
            for sensor, pending in telemetry.items():
                while pending and pending[0][0] <= now:
                    _, arrival, data = pending.popleft()
                    states[f"budgeted_{sensor}"] = (arrival, data)
            if message is not None:
                kind = message.get_type()
                if kind == "HEARTBEAT":
                    armed = bool(message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                    if not stream_setup:
                        att_rate = 15 if args.downlink_bps else 30
                        range_rate = 10 if args.downlink_bps else 30
                        for msg_id, rate in [(108, 30), (30, att_rate), (132, range_rate), (193, 5), (32, 10)]:
                            request_message(mav, msg_id, rate)
                        stream_setup = True
                        received_since = now
                if kind == "STATUSTEXT":
                    statuses.append(message.text)
                    trace.write(json.dumps({"t": now, "status": message.text}) + "\n")
                if kind == "COMMAND_ACK":
                    trace.write(json.dumps({"t": now, "ack": message.command, "result": message.result}) + "\n")

            if received_since is not None and now - received_since > 13 and now - last_control > 2:
                if not armed and armed_at is None:
                    mav.mav.set_mode_send(1, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4)
                    command(mav, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1, 21196)
                    phase = "arming"
                elif armed and armed_at is None:
                    command(mav, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 1.4)
                    armed_at = now
                    phase = "takeoff"
                last_control = now

            if armed_at is not None and now - armed_at > 5 and "SIM_STATE" in states:
                sample_t, state = states["SIM_STATE"]
                if origin is None:
                    origin = state.lat_int * 1e-7, state.lon_int * 1e-7, state.alt - 1.4
                    # Fly a slow lateral pass to seed the metric landmark map.
                    phase = "translate"
                p_local = local_position(state, origin)
                if start_pos is None and "LOCAL_POSITION_NED" in states:
                    base = states["LOCAL_POSITION_NED"][1]
                    start_pos = np.array([base.x, base.y, base.z]) - p_local
                if phase == "translate" and now - next_send > 0.15:
                    speed = 0.27 if switched_at is None else 0.0
                    mav.mav.set_position_target_local_ned_send(
                        int(now * 1000), 1, 1, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                        0x0DC7, 0, 0, 0, speed, 0, 0, 0, 0, 0, 0, 0)
                    next_send = now
                if sample_t > last_capture:
                    last_capture = sample_t
                    if switched_at is not None and landing_at is None and sample_t >= switched_at + 1:
                        hold_truth.append(p_local[:2])
                    attitude = (state.roll, state.pitch, state.yaw)
                    body = euler_to_rot(*attitude)
                    room_pos = p_local + np.array([3, 3, 0])
                    image = render(room, camera, body @ camera_to_body(camera.uptilt_deg),
                                   room_pos + body @ np.asarray(mount.camera_B), rng=rng)
                    if args.exposure_ms:
                        dx, dy = state.zgyro, state.ygyro
                        length = min(31, max(args.blur_min_px, round(
                            camera.fx * math.hypot(dx, dy) * args.exposure_ms / 1000)))
                        blur_lengths.append(length)
                        image = motion_blur(image, length, math.atan2(dy, dx))
                    blackout = (args.blackout_seconds > 0 and switched_at is not None
                                and 2 <= now - switched_at < 2 + args.blackout_seconds)
                    if blackout or len(queue) == queue.maxlen or rng.random() < args.drop:
                        dropped += 1
                    else:
                        delay = max(0, args.delay_ms + rng.normal(0, args.jitter_ms)) / 1000
                        queue.append((sample_t + delay, sample_t, image, p_local, attitude))
                        queue = deque(sorted(queue, key=lambda item: item[0]), maxlen=queue.maxlen)
                        peak_queue = max(peak_queue, len(queue))
                if queue and queue[0][0] <= now:
                    _, capture, image, truth, truth_att = queue.popleft()
                    # ArduPilot's real telemetry and frames share the receive clock here.
                    att_entry = states.get("budgeted_ATTITUDE")
                    range_entry = states.get("budgeted_DISTANCE_SENSOR")
                    if (capture <= last_processed or now - capture > 0.2 or att_entry is None or range_entry is None
                            or now - min(att_entry[0], range_entry[0]) > 0.2):
                        dropped += 1
                        continue
                    last_processed = capture
                    attitude = tuple(getattr(att_entry[1], axis) for axis in ("roll", "pitch", "yaw"))
                    distance = range_entry[1].current_distance / 100
                    started = time.monotonic()
                    out = pipe.step(image, capture, distance, attitude)
                    process_ms = (time.monotonic() - started) * 1000
                    frames += 1
                    valid += int(out.valid)
                    pose = None
                    if first_truth is None:
                        first_truth = truth.copy()
                    if start_pos is not None and (args.fixture or out.valid):
                        pose = start_pos + (truth if args.fixture else out.p_WB + first_truth)
                    if switched_at is not None and landing_at is None:
                        hold.append((capture, truth[:2], (out.p_WB + first_truth - truth)[:2] if out.valid else None,
                                     states.get("LOCAL_POSITION_NED", (None, None))[1]))
                    if pose is not None and now >= last_pose_send:
                        if args.outlier_m and switched_at is not None and now - switched_at > 1 and not injected_outlier:
                            pose = pose + np.array([args.outlier_m, 0, 0])
                            injected_outlier = True
                            outlier_target = (now, pose.copy(), pose[0] - args.outlier_m)
                            outlier_usec = int((epoch + now) * 1e6)
                            trace.write(json.dumps({"t": now, "event": "external pose outlier",
                                                    "offset_m": args.outlier_m}) + "\n")
                        msg = vision_position_estimate(epoch + now, pose, truth_att if args.fixture else out.rpy,
                                                       out.pos_sigma_m, out.reset_counter, mav.mav,
                                                       compact=not args.full_covariance)
                        size = len(msg.pack(mav.mav))
                        mav.mav.send(msg)
                        last_pose_send = now + max(interval, size / args.uplink_bps)
                        sent += 1
                        if switched_at is not None and landing_at is None and not fallback_sent:
                            hold_poses += 1
                            hold_pose_gaps.append(now - (last_hold_pose if last_hold_pose is not None else switched_at))
                            last_hold_pose = now
                        pose_bytes += size
                        if sent >= 20 and switched_at is None:
                            command(mav, mavutil.mavlink.MAV_CMD_SET_EKF_SOURCE_SET, 2)
                            switched_at = now
                    record = {"att": attitude, "range_m": distance, "valid": out.valid, "status": out.status,
                              "p": out.p_WB.tolist(), "rpy": out.rpy, "truth_p": truth.tolist(),
                              "truth_rpy": truth_att, "reset": out.reset_counter, "processing_ms": process_ms,
                              "pose_count": sent, "pose_bytes": pose_bytes, "frame_age_ms": (now - capture) * 1000}
                    recorder.write(capture, image, record)
                    trace.write(json.dumps({"t": capture, **record}) + "\n")
                if (switched_at is not None and args.blackout_seconds > 0 and not fallback_sent
                        and now - switched_at >= 2.4):
                    command(mav, mavutil.mavlink.MAV_CMD_SET_EKF_SOURCE_SET, 3 if args.fallback_source == "flow" else 1)
                    fallback_sent = True
                    trace.write(json.dumps({"t": now, "event": "simulated pilot source switch"}) + "\n")
                if switched_at is not None and landing_at is None and now - switched_at >= args.hold_seconds:
                    mav.mav.set_mode_send(1, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9)
                    phase = "land-commanded"
                    landing_at = now
                if landing_at is not None and not armed:
                    phase = "landed-and-disarmed"
                    break

    if switched_at is not None:
        active_end = min(switched_at + (2.0 if args.blackout_seconds else args.hold_seconds), now)
        hold_pose_gaps.append(max(0, active_end - (last_hold_pose or switched_at)))
    metrics = hold_metrics(hold, switched_at, args.fixture)
    if hold_truth:
        metrics["drift_m"] = float(np.max(np.linalg.norm(np.array(hold_truth) - hold_truth[0], axis=1)))
        metrics["truth_samples"] = len(hold_truth)
    return {"mode": "truth fixture" if args.fixture else "live kx-vo", "tracker": args.tracker,
            "checkpoint_sha256": tracker.checkpoint_hash if tracker else None, "phase": phase,
            "frames": frames, "valid_frames": valid, "dropped": dropped, "poses_sent": sent, "pose_bytes": pose_bytes,
            "outlier_injected": injected_outlier, "downlink_dropped": downlink_dropped, "peak_frame_queue": peak_queue, "fallback_commanded": fallback_sent,
            "statuses": statuses, "hold": metrics,
            "hold_poses": hold_poses, "max_hold_pose_gap_s": max(hold_pose_gaps, default=None),
            "external_nav_hold": hold_metrics(
                [row for row in hold if not args.blackout_seconds or row[0] < switched_at + 2],
                switched_at, args.fixture),
            "blur_frames_ge_3px": sum(n >= 3 for n in blur_lengths),
            "outlier_response": outlier_response(hold, outlier_target), "outlier_usec": outlier_usec}


def hold_metrics(rows, switched_at, fixture):
    if switched_at is None:
        return {}
    settled = [row for row in rows if row[0] >= switched_at + 1]
    if not settled:
        return {}
    points = np.array([row[1] for row in settled])
    errors = [row[2] for row in settled if row[2] is not None]
    return {"drift_m": float(np.max(np.linalg.norm(points - points[0], axis=1))),
            "rms_error_m": 0.0 if fixture else (float(np.sqrt(np.mean(np.sum(np.square(errors), axis=1)))) if errors else None),
            "valid_fraction": 1.0 if fixture else len(errors) / len(settled),
            "duration_s": settled[-1][0] - settled[0][0]}


def outlier_response(rows, target):
    if target is None:
        return None
    t, injected, nominal_north = target
    samples = [row[3].x for row in rows if t < row[0] < t + 0.8 and row[3] is not None]
    peak = max((abs(x - nominal_north) for x in samples), default=None)
    return {"injected_north_m": float(injected[0]), "peak_ekf_deviation_from_nominal_m": peak,
            "no_large_state_jump_observed": None if peak is None else bool(peak < abs(injected[0] - nominal_north) / 2)}


def acceptance_failures(result, args):
    failures = []
    evidence = result["dataflash"]
    for key in ("secondary_samples", "vision_accepted", "secondary_horizontal_innovation_samples"):
        if not evidence.get(key):
            failures.append(f"missing EKF evidence: {key}")
    if result["phase"] != "landed-and-disarmed":
        failures.append("landing/disarm unconfirmed")
    hold = result["hold"]
    if result["hold_poses"] < 8 or (result.get("max_hold_pose_gap_s") or 0) > 0.5:
        failures.append("insufficient pose stream during ExternalNav hold")
    if hold.get("duration_s", 0) < args.hold_seconds - 2:
        failures.append("insufficient hold duration")
    if hold.get("drift_m", math.inf) > args.max_hold_drift_m:
        failures.append("hold drift exceeds limit")
    active = result.get("external_nav_hold", hold)
    if active.get("rms_error_m") is None or active["rms_error_m"] > args.max_hold_error_m:
        failures.append("horizontal estimator RMS exceeds limit")
    if active.get("valid_fraction", 0) < 0.9:
        failures.append("insufficient valid estimate coverage")
    if getattr(args, "blur_min_px", 0) >= 3 and args.exposure_ms and not result["blur_frames_ge_3px"]:
        failures.append("requested blur was not applied")
    if args.blackout_seconds and not result["fallback_commanded"]:
        failures.append("fallback switch missing")
    if args.blackout_seconds and args.fallback_source == "flow" and not any(
            "fusing optical flow" in status.lower() for status in result["statuses"]):
        failures.append("optical flow fusion unconfirmed")
    if args.outlier_m:
        if not evidence.get("outlier_position_gate_exceeded"):
            failures.append("outlier rejection unconfirmed in EKF innovation logs")
        response = result["outlier_response"]
        peak = response["peak_ekf_deviation_from_nominal_m"] if response else None
        if peak is None or peak > args.max_hold_error_m:
            failures.append("EKF response to outlier exceeded limit or was unobserved")
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arducopter", type=Path, required=True)
    parser.add_argument("--params", type=Path, default=Path("params/sitl-kxvo.parm"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracker", choices=("klt", "orb", "xfeat"), default="klt")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--xfeat-source", type=Path, default=Path("runs/xfeat-upstream"))
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--outlier-m", type=float, default=0,
                        help="Inject one north-position jump after switching to ExternalNav")
    parser.add_argument("--full-covariance", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=85)
    parser.add_argument("--hold-seconds", type=float, default=8)
    parser.add_argument("--max-hold-drift-m", type=float, default=0.5)
    parser.add_argument("--max-hold-error-m", type=float, default=0.3)
    parser.add_argument("--delay-ms", type=float, default=0)
    parser.add_argument("--fallback-source", choices=("gps", "flow"), default="gps")
    parser.add_argument("--viso-delay-ms", type=int, help="Override the nominal injected delay compensation")
    parser.add_argument("--jitter-ms", type=float, default=0)
    parser.add_argument("--drop", type=float, default=0)
    parser.add_argument("--uplink-bps", type=float, default=1250)
    parser.add_argument("--downlink-bps", type=float, default=0, help="Limit estimator telemetry bytes/s; zero means unlimited")
    parser.add_argument("--exposure-ms", type=float, default=0)
    parser.add_argument("--blur-min-px", type=int, default=0, help="Explicit synthetic blur floor when exposure is enabled")
    parser.add_argument("--blackout-seconds", type=float, default=0)
    args = parser.parse_args(argv)
    if not args.arducopter.is_file() or not args.params.is_file():
        parser.error("Pass the local native ArduCopter binary and a SITL parameter file")
    if not 0 <= args.drop < 1 or args.uplink_bps <= 0:
        parser.error("Use a drop probability in [0, 1) and a positive uplink budget")
    if min(args.delay_ms, args.jitter_ms, args.exposure_ms, args.blackout_seconds, args.downlink_bps) < 0:
        parser.error("Delay, jitter, exposure and blackout must be nonnegative")
    if args.blackout_seconds and args.hold_seconds <= 2 + args.blackout_seconds:
        parser.error("Hold duration must extend past the blackout")
    args.output.mkdir(parents=True, exist_ok=False)
    compensation = round(args.delay_ms) if args.viso_delay_ms is None else args.viso_delay_ms
    if not 0 <= compensation <= 250:
        parser.error("VISO delay compensation must be in [0, 250] ms")
    overrides = args.params.read_text() + f"\nVISO_DELAY_MS {compensation}\n"
    if args.fallback_source == "flow":
        overrides += "SIM_FLOW_ENABLE 1\nFLOW_TYPE 10\nEK3_SRC3_POSXY 0\nEK3_SRC3_VELXY 5\nEK3_SRC3_POSZ 1\nEK3_SRC3_VELZ 0\nEK3_SRC3_YAW 1\n"
    effective_params = args.output / "effective.parm"
    effective_params.write_text(overrides)
    port = free_port()
    # No user-supplied MAVLink endpoint. The runner only talks to its child simulator.
    link = mavutil.mavlink_connection(f"udpin:127.0.0.1:{port}", source_system=255)
    command_line = [str(args.arducopter.resolve()), "--model", "+", "--speedup", "1",
                    "--home", "51.5,-0.1,50,0", "--serial0", f"udpclient:127.0.0.1:{port}",
                    "--defaults", str(effective_params.resolve()), "-I", str(args.instance), "-w"]
    (args.output / "setup.json").write_text(json.dumps({"command": command_line,
        "binary_sha256": sha256(args.arducopter), "params_sha256": sha256(effective_params), "effective_viso_delay_ms": compensation,
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, indent=2) + "\n")
    recorder = Recorder(args.output / "recording")
    try:
        with (args.output / "arducopter.log").open("w") as console:
            child = subprocess.Popen(command_line, cwd=args.output, stdout=console, stderr=subprocess.STDOUT)
            try:
                result = run(args, link, recorder)
            finally:
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
                    try:
                        child.wait(timeout=4)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
    finally:
        link.close()
        recorder.close()
    logs = sorted((args.output / "logs").glob("*.BIN"))
    result["dataflash"] = ekf_evidence(logs[-1], result["outlier_usec"], args.outlier_m) if logs else {}
    result["acceptance_failures"] = acceptance_failures(result, args)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return int(bool(result["acceptance_failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
