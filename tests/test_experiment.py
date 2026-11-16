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


def test_manifest_rejects_changed_pinned_image(tmp_path):
    (tmp_path / "a.png").write_bytes(b"changed")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,split,group,rights,sha256\na.png,train,one,CC BY 4.0,wrong\n")
    with pytest.raises(ValueError, match="Image hash does not match manifest"):
        read_manifest(manifest)


def test_tracking_health_counts_unavailable_time_without_reference():
    from kxvo.score import tracking_health

    rows = [{"t": t, "valid": valid, "reset": reset, "processing_ms": 4}
            for t, valid, reset in [(0, False, 0), (0.2, True, 0), (0.5, False, 1),
                                    (0.9, False, 1), (1.5, True, 1), (2, True, 1)]]
    metrics = tracking_health(rows, max_frame_gap=1.0)
    assert metrics["valid_frames"] == 3
    assert metrics["initialization_s"] == pytest.approx(0.2)
    assert metrics["unavailable_s"] == pytest.approx(1.2)
    assert metrics["longest_unavailable_s"] == pytest.approx(1.0)
    assert metrics["reset_changes"] == 1


def test_tracking_health_counts_gap_between_valid_frames():
    from kxvo.score import tracking_health
    rows = [{"t": t, "valid": True} for t in [0, 0.05, 0.9, 0.95]]
    metrics = tracking_health(rows, max_frame_gap=0.2)
    assert metrics["outage_count"] == 1
    assert metrics["longest_unavailable_s"] == pytest.approx(0.85)


def test_warp_stream_is_shared_between_clean_and_blur():
    image = np.random.default_rng(9).integers(256, size=(64, 64), dtype=np.uint8)
    clean_rng = np.random.default_rng(10)
    degraded_rng = np.random.default_rng(10)
    blur_rng = np.random.default_rng(11)
    for _ in range(4):
        _, _, clean_transform = pair(image, clean_rng, False, 11)
        _, _, degraded_transform = pair(image, degraded_rng, True, 11, blur_rng=blur_rng)
        np.testing.assert_array_equal(clean_transform, degraded_transform)


def test_held_out_cli_accepts_test_only_manifest(tmp_path, monkeypatch):
    from kxvo import xfeat_eval
    cv2.imwrite(str(tmp_path / "frame.png"), np.zeros((32, 32), dtype=np.uint8))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,split,group,rights\nframe.png,test,separate,own\n")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "scores.json"
    monkeypatch.setattr(xfeat_eval, "load_xfeat", lambda *args: object())
    monkeypatch.setattr(xfeat_eval, "evaluate_pairs", lambda *args, **kwargs: {
        "clean": {"recall": 0.5}, "blur": {"recall": 0.4},
    })
    xfeat_eval.main(["--manifest", str(manifest), "--checkpoint", str(checkpoint), "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["frames"] == 1
    assert result["groups"] == ["separate"]


def test_archive_member_resume_reuses_only_crc_verified_files(tmp_path):
    from io import BytesIO
    from zipfile import ZipFile
    from kxvo.tartanair import extract_members

    payload = BytesIO()
    with ZipFile(payload, "w") as archive:
        archive.writestr("one.png", b"one")
        archive.writestr("two.png", b"two")

    class InterruptedArchive(BytesIO):
        size = len(payload.getvalue())
        transferred = 0
        attempts = 0

        def fetch(self, start, end):
            self.attempts += 1
            if self.attempts == 2:
                raise OSError("interrupted")
            return self.getvalue()[start:end + 1], start, self.size

    remote = InterruptedArchive(payload.getvalue())
    with ZipFile(remote) as archive:
        entries = [(info, tmp_path / info.filename) for info in archive.infolist()]
        with pytest.raises(OSError, match="interrupted"):
            extract_members(remote, archive, entries, 1)
        assert (tmp_path / "one.png").read_bytes() == b"one"
        assert not (tmp_path / "two.png").exists()
        assert len(extract_members(remote, archive, entries, 1)) == 2
        assert remote.attempts == 3
        (tmp_path / "two.png").write_bytes(b"bad")
        extract_members(remote, archive, entries, 1)
        assert remote.attempts == 4
        assert (tmp_path / "two.png").read_bytes() == b"two"


def test_reliability_distillation_tracks_teacher_and_masks_warped_edges():
    torch = pytest.importorskip("torch")
    from kxvo.train import reliability_loss

    predicted = torch.full((1, 2, 3), 0.5, requires_grad=True)
    teacher = np.array([[0.2, 0.8, 0.9], [0.2, 0.8, 0.9]], dtype=np.float32)
    inverse = np.array([[1., 0., -8.], [0., 1., 0.], [0., 0., 1.]])
    reliability_loss(predicted, teacher, inverse).backward()
    assert (predicted.grad[:, :, 0] == 0).all()
    assert (predicted.grad[:, :, 1] > 0).all()
    assert (predicted.grad[:, :, 2] < 0).all()

def test_teacher_requests_all_detections_for_negative_mask():
    torch = pytest.importorskip("torch")
    from kxvo.train import detector_targets

    points = np.column_stack(np.unravel_index(np.arange(400), (64, 64))).astype(np.float32)

    class Teacher:
        def detectAndCompute(self, clean, top_k):
            return [{"keypoints": torch.from_numpy(points[:top_k])}]

    first, second = detector_targets(Teacher(), np.zeros((64, 64), dtype=np.uint8), np.eye(3))
    np.testing.assert_array_equal(first, points)
    np.testing.assert_array_equal(second, points)


def test_geometric_projection_rejects_an_occluded_surface():
    from kxvo.geometric_pairs import Projection

    depth = np.full((64, 64), 4.0, dtype=np.float32)
    other = depth.copy()
    other[16, 27] = 1.0
    projection = Projection(np.array([20., 20.]), np.array([32., 32.]), np.ones(2),
                            depth, other, np.eye(3), np.array([-1., 0., 0.]))
    points = np.array([[32., 16.], [32., 32.]])
    targets, visible = projection.visible(points)
    np.testing.assert_array_equal(targets, [[27., 16.], [27., 32.]])
    np.testing.assert_array_equal(visible, [False, True])


def test_lockstep_servo_packets_check_length_and_magic():
    import struct
    from kxvo.lockstep import servo_frame

    sixteen = struct.pack("<HHI16H", 18458, 400, 123, *([1000] * 16))
    thirty_two = struct.pack("<HHI32H", 29569, 400, 124, *([1000] * 32))
    assert servo_frame(sixteen) == (400, 123)
    assert servo_frame(thirty_two) == (400, 124)
    with pytest.raises(ValueError):
        servo_frame(thirty_two[:40])


def test_paired_windows_use_the_same_errors_and_report_lost_windows():
    from kxvo.score import compare_windows

    t = np.arange(0, 12, 0.1)
    truth = np.column_stack((t, t, t * 0, t * 0, t * 0))
    full = [{"t": time, "p": [time, 0, 0], "rpy": [0, 0, 0], "valid": True} for time in t]
    failed = [{**row, "valid": row["t"] != 4.0} for row in full]
    traces = {"full": {}, "gaps": {}}
    first, _ = score(full, truth, window_details=traces["full"])
    second, _ = score(failed, truth, window_details=traces["gaps"])
    paired = compare_windows(traces)
    assert paired["eligible"] == first["relative_windows_eligible"] == second["relative_windows_eligible"]
    assert paired["common"] == second["relative_pairs"] < paired["eligible"]
    assert paired["arms"]["full"]["lost"] == 0
    assert paired["arms"]["gaps"]["lost"] == paired["eligible"] - paired["common"]
    assert paired["arms"]["gaps"]["median_translation_m"] == pytest.approx(0)


def test_frozen_manifest_checks_the_csv_before_reading_images(tmp_path):
    from kxvo.trackers import sha256
    from kxvo.train import verify_manifest

    path = tmp_path / "manifest.csv"
    path.write_text("first")
    expected = sha256(path)
    assert verify_manifest(path, expected) == expected
    path.write_text("revised")
    with pytest.raises(ValueError, match="Manifest SHA-256 mismatch"):
        verify_manifest(path, expected)


def test_recorded_pose_stream_age_and_validation(tmp_path):
    from kxvo.pose_stream import PoseStream

    path = tmp_path / "poses.jsonl"
    path.write_text('{"t":0.1,"p":[1,2,3],"rpy":[0,0,0]}\n'
                    '{"t":0.2,"p":[2,2,3],"rpy":[0,0,0],"valid":false}\n')
    stream = PoseStream(path)
    assert stream.sample(0) == (None, None)
    assert stream.sample(0.15)[1].p_WB.tolist() == [1, 2, 3]
    assert not stream.sample(0.21)[1].valid
    assert stream.sample(0.5) == (None, None)
    with path.open("a") as output:
        output.write('{"t":0.2,"p":[1,2,3],"rpy":[0,0,0]}\n')
    with pytest.raises(ValueError, match="Invalid pose stream row 3"):
        PoseStream(path)
