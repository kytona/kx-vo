import math

import numpy as np
import pytest
from pymavlink.dialects.v20 import ardupilotmega as mavlink2

from kxvo.bench import run_synth
from kxvo.camera import Camera
from kxvo.geometry import camera_to_body, euler_to_rot, rot_to_euler
from kxvo.mav import covariance, vision_position_estimate
from kxvo.pipeline import Mount, Pipeline
from kxvo.synth import Room, default_camera, simulate
from kxvo.video import Recorder, replay_records


# ------------------------------------------------------------------ geometry
def test_euler_round_trip():
    for rpy in [(0.1, -0.2, 2.5), (-0.4, 0.3, -1.0), (0.0, 0.0, 0.0)]:
        back = rot_to_euler(euler_to_rot(*rpy))
        assert np.allclose(back, rpy, atol=1e-12)


def test_camera_axes_in_body():
    R = camera_to_body(0.0)
    assert np.allclose(R @ [0, 0, 1], [1, 0, 0])  # optical axis forward
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])  # image right = body right
    assert np.allclose(R @ [0, 1, 0], [0, 0, 1])  # image down = body down
    up = camera_to_body(20.0) @ [0, 0, 1]
    assert up[2] < 0 and math.isclose(math.degrees(math.asin(-up[2])), 20.0, abs_tol=1e-9)


def test_camera_normalize_and_scale():
    cam = Camera(width=640, height=480, fx=400, fy=400, cx=320, cy=240, mask=[[0, 0, 100, 20]])
    x = cam.normalize(np.array([[320.0, 240.0], [720.0, 240.0]]))
    assert np.allclose(x, [[0, 0], [1, 0]])
    half = cam.scaled(320)
    assert (half.width, half.height, half.fx, half.mask) == (320, 240, 200, [[0, 0, 50, 10]])
    m = half.feature_mask()
    assert m[5, 20] == 0 and m[100, 100] == 255
    assert cam.normalize(np.empty((0, 2))).shape == (0, 2)


def test_camera_height_from_slant_range():
    pipe = Pipeline(default_camera(), Mount(camera_B=(0.0, 0.0, 0.0), rangefinder_B=(0.0, 0.0, 0.0)))
    R = euler_to_rot(math.radians(10), math.radians(-20), 0.3)
    slant = 1.0 / (math.cos(math.radians(10)) * math.cos(math.radians(-20)))
    assert math.isclose(pipe.camera_height(slant, R), 1.0, rel_tol=1e-9)
    assert pipe.camera_height(None, R) is None


# ------------------------------------------------------------------- mavlink
def test_vision_position_estimate_round_trip():
    mav = mavlink2.MAVLink(None, srcSystem=1, srcComponent=197)
    msg = vision_position_estimate(12.5, [1.0, -2.0, -0.8], (0.01, -0.02, 1.5), 0.05, 3, mav)
    buf = msg.pack(mav)
    parser = mavlink2.MAVLink(None)
    decoded = parser.parse_char(buf)
    assert decoded.get_type() == "VISION_POSITION_ESTIMATE"
    assert decoded.usec == 12_500_000
    assert np.allclose([decoded.x, decoded.y, decoded.z], [1.0, -2.0, -0.8], atol=1e-6)
    assert np.allclose([decoded.roll, decoded.pitch, decoded.yaw], [0.01, -0.02, 1.5], atol=1e-6)
    assert decoded.reset_counter == 3
    assert math.isclose(decoded.covariance[0], 0.05**2, rel_tol=1e-5)


def test_compact_message_is_small():
    mav = mavlink2.MAVLink(None, srcSystem=1, srcComponent=197)
    full = vision_position_estimate(1.0, [1, 2, 3], (0, 0, 0), 0.05, 0, mav).pack(mav)
    small = vision_position_estimate(1.0, [1, 2, 3], (0, 0, 0), 0.05, 0, mav, compact=True).pack(mav)
    assert len(small) < 50 < 120 < len(full)
    decoded = mavlink2.MAVLink(None).parse_char(small)
    assert decoded.x == pytest.approx(1.0) and decoded.covariance[0] == 0.0


def test_covariance_layout():
    cov = covariance(0.1, 0.2)
    assert len(cov) == 21
    assert [cov[i] for i in (0, 6, 11)] == pytest.approx([0.01] * 3)
    assert [cov[i] for i in (15, 18, 20)] == pytest.approx([0.04] * 3)
    assert sum(cov) == pytest.approx(0.03 + 0.12)


# ------------------------------------------------------------------ synthetic
def test_floor_landmarks_exact_with_perfect_sensors():
    cam, room = default_camera(), Room()
    mount = Mount(camera_B=(0.03, 0.0, 0.0), rangefinder_B=(0.0, 0.0, 0.0))
    pipe = Pipeline(cam, mount)
    s0 = next(simulate(duration=0.1, cam=cam, room=room, cam_offset_B=mount.camera_B,
                       range_sigma=0, att_sigma_deg=0, yaw_drift_dps=0))
    pipe.step(s0.image, s0.t, s0.range_m, s0.att_meas)
    vo = pipe.vo
    floor = np.isfinite(vo.lm[:, 0]) & vo.floor
    assert floor.sum() >= 20
    # VO's world origin is the start body position, so the floor (true z = 0)
    # sits at z = -p_WB0.z.
    assert np.allclose(vo.lm[floor, 2], -s0.p_WB[2], atol=1e-6)


def test_mavlink_loopback_telemetry_and_pose():
    import socket
    import time

    from pymavlink import mavutil

    from kxvo.mav import MavLink

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    vo_link = MavLink(f"udpin:127.0.0.1:{port}")
    fc = mavutil.mavlink_connection(f"udpout:127.0.0.1:{port}", source_system=1, source_component=1)
    try:
        deadline = time.monotonic() + 3.0
        att = rng = None
        while time.monotonic() < deadline and (att is None or rng is None):
            fc.mav.heartbeat_send(2, 3, 0, 0, 0)
            fc.mav.attitude_send(0, 0.1, -0.05, 1.2, 0, 0, 0)
            fc.mav.distance_sensor_send(0, 2, 600, 95, 0, 0, 25, 0)
            time.sleep(0.05)
            att, rng = vo_link.snapshot()
        assert att == pytest.approx((0.1, -0.05, 1.2), abs=1e-6)
        assert rng == pytest.approx(0.95)
        vo_link.send_pose(1.0, [0.5, 0.25, -1.0], (0.1, -0.05, 1.2), 0.05, 0)
        got = fc.recv_match(type="VISION_POSITION_ESTIMATE", blocking=True, timeout=2.0)
        assert got is not None and got.x == pytest.approx(0.5) and got.get_srcComponent() == 197
    finally:
        vo_link.close()
        fc.close()


@pytest.mark.parametrize("seed", [1, 3])
def test_synthetic_loop_accuracy(seed):
    r = run_synth(duration=24.0, seed=seed)
    assert r.valid_ratio > 0.97
    assert r.path_m > 8.0
    # Regression guard, set above measured values (seed 1 about 0.07 m, seed 3
    # about 0.06 m horizontal RMSE). Synthetic best case, not flight evidence.
    assert r.rmse_m < 0.2
    assert r.yaw_rmse_deg < 3.0


def test_blank_video_publishes_nothing():
    cam = default_camera()
    pipe = Pipeline(cam)
    blank = np.full((cam.height, cam.width), 128, np.uint8)
    outs = [pipe.step(blank, i / 30, 1.0, (0.0, 0.0, 0.0)) for i in range(10)]
    assert not any(o.valid for o in outs)


def test_record_and_replay_match(tmp_path):
    cam = default_camera()
    mount = Mount(camera_B=(0.03, 0.0, 0.0), rangefinder_B=(0.0, 0.0, 0.0))
    live = Pipeline(cam, mount)
    rec = Recorder(tmp_path)
    live_out = []
    for s in simulate(duration=2.0, cam=cam, cam_offset_B=mount.camera_B):
        o = live.step(s.image, s.t, s.range_m, s.att_meas)
        live_out.append(o.p_WB.copy())
        rec.write(s.t, s.image, {"range_m": s.range_m, "att": list(s.att_meas)})
    rec.close()
    again = Pipeline(cam, mount)
    replay_out = [again.step(g, t, r, a).p_WB.copy() for t, g, r, a in replay_records(tmp_path)]
    assert len(replay_out) == len(live_out)
    assert np.allclose(replay_out[-1], live_out[-1], atol=1e-6)
