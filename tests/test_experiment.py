import json

import cv2
import numpy as np
import pytest

from kxvo.camera import Camera
from kxvo.score import score
from kxvo.trackers import DescriptorTracker
from kxvo.train import pair, project, read_manifest
from kxvo.vo import MonoVO, VOConfig


def test_orb_recovers_verified_landmark_and_expires_cache():
    cam = Camera(160, 128, 90, 90, 80, 64)
    tracker = DescriptorTracker("orb", 80)
    vo = MonoVO(cam, VOConfig(max_corners=80), tracker)
    image = np.random.default_rng(5).integers(0, 255, (128, 160), dtype=np.uint8)
    vo._detect(image)
    assert len(vo.pts) > 10
    vo.lm[:] = np.column_stack((cam.normalize(vo.pts), np.ones(len(vo.pts))))
    vo.verified[:] = True
    vo.t_last = 1
    assert vo._descriptor_track(np.zeros_like(image), 1.1)["recovered"] == 0
    assert len(vo.lost_tracks) > 10
    assert vo._descriptor_track(image.copy(), 1.2)["recovered"] > 10
    vo.t_last = 1.2
    vo._descriptor_track(np.zeros_like(image), 1.3)
    assert vo._descriptor_track(image.copy(), 1.8)["recovered"] == 0
    assert not vo.lost_tracks


def test_descriptor_matching_rejects_ambiguous_candidates():
    tracker = DescriptorTracker("orb", 20)
    desc = np.zeros((2, 32), np.uint8)
    assert tracker.matches(desc[:1], desc) == []


def test_pair_is_deterministic_and_maps_pixels():
    image = np.zeros((96, 128), np.uint8)
    image[48, 64] = 255
    _, warped, matrix = pair(image, np.random.default_rng(6), False, 9)
    a = project([[64, 48]], matrix)[0]
    y, x = np.unravel_index(warped.argmax(), warped.shape)
    assert np.linalg.norm(a - [x, y]) < 1


def test_manifest_rejects_cross_split_duplicate(tmp_path):
    image = np.zeros((96, 128), np.uint8)
    cv2.imwrite(str(tmp_path / "a.png"), image)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,split,group,rights\na.png,train,1,own\na.png,validation,2,own\n")
    with pytest.raises(ValueError, match="Identical image crosses splits"):
        read_manifest(manifest)


def test_metric_scale_stays_fixed_and_yaw_wraps():
    t = np.arange(0, 8, 0.1)
    ref = np.column_stack((t, t, t * 0, t * 0, t * 0 + np.pi - 0.01))
    rows = [{"t": ts, "p": [2 * ts, 0, 0], "rpy": [0, 0, -np.pi - 0.01],
             "valid": True} for ts in t]
    metrics, _ = score(rows, ref)
    assert metrics["yaw_rms_deg"] < 1e-6
    assert metrics["relative_translation_m_median"] == pytest.approx(5.0, abs=0.1)
    json.dumps(metrics, allow_nan=False)


def test_compact_wire_size_after_reset():
    from kxvo.mav import mavutil, vision_position_estimate
    mav = mavutil.mavlink.MAVLink(None)
    args = (1.0, (1.0, 2.0, -1.0), (0.1, 0.2, 0.3), 0.2)
    small = vision_position_estimate(*args, 0, mav, compact=True)
    reset = vision_position_estimate(*args, 1, mav, compact=True)
    assert len(small.pack(mav)) == 44
    assert len(reset.pack(mav)) == 129


def test_pose_publication_rejects_stale_or_missing_inputs():
    from kxvo.cli import can_publish_pose
    assert can_publish_pose(True, 1, (0, 0, 0), 1, 1.19, 0.2)
    assert not can_publish_pose(True, 1, (0, 0, 0), 1, 1.21, 0.2)
    assert not can_publish_pose(True, 1, None, 1, 1.01, 0.2)
    assert not can_publish_pose(True, 0, (0, 0, 0), 1, 1.01, 0.2)


def test_sitl_uses_integer_geography():
    from types import SimpleNamespace
    from kxvo.sitl import local_position
    state = SimpleNamespace(lat=515000000.0, lon=-1000000.0, lat_int=515000001,
                            lon_int=-999999, alt=51.4)
    p = local_position(state, (51.5, -0.1, 50))
    assert p[0] == pytest.approx(0.01113195)
    assert p[1] > 0
    assert p[2] == pytest.approx(-1.4)


def test_sitl_hold_requires_pose_stream_and_low_drift():
    from types import SimpleNamespace
    from kxvo.sitl import acceptance_failures
    args = SimpleNamespace(hold_seconds=8, max_hold_drift_m=0.5, max_hold_error_m=0.3,
                           blackout_seconds=0, outlier_m=0, fallback_source="gps")
    result = {"dataflash": {"secondary_samples": 10, "vision_accepted": 10,
                            "secondary_horizontal_innovation_samples": 10},
              "phase": "landed-and-disarmed", "hold_poses": 0,
              "hold": {"duration_s": 7, "drift_m": 3, "rms_error_m": 0.02, "valid_fraction": 1}}
    assert len(acceptance_failures(result, args)) == 2


@pytest.mark.parametrize("frame_format", ["png", "npy"])
def test_recording_keeps_full_color_and_replay_selects_width(tmp_path, frame_format):
    from kxvo.video import Recorder, replay_records
    image = np.zeros((240, 320, 3), np.uint8)
    image[:, :, 2] = 200
    run = tmp_path / "capture"
    rec = Recorder(run, frame_format=frame_format)
    rec.write(0, image, {"att": [0, 0, 0], "range_m": 1})
    rec.close()
    frame_path = run / f"frames/000000.{frame_format}"
    stored = np.load(frame_path, allow_pickle=False) if frame_format == "npy" else cv2.imread(str(frame_path))
    assert np.array_equal(stored, image)
    t, gray, distance, attitude = next(replay_records(run, 160))
    assert gray.shape == (120, 160)
    assert (t, distance, attitude) == (0, 1, (0, 0, 0))


def test_descriptor_geometry_rejects_large_jump_with_few_matches():
    cam = Camera(320, 240, 150, 150, 160, 120)
    vo = MonoVO(cam)
    vo.pts = np.array([[100, 100], [105, 100]], dtype=np.float32)
    matches = vo._geometric_matches([(0, 0), (1, 1)], np.array([[103, 100], [220, 100]]))
    assert matches == [(0, 0)]
