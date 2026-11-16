"""Depth and pose supervision for adjacent TartanAir V2 pinhole frames."""

import csv
from dataclasses import dataclass
import json
from functools import cached_property

import cv2
import numpy as np

from .trackers import sha256

# OpenCV right/down/forward coordinates expressed in the pose's forward/right/down axes.
FRD_FROM_CAMERA = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])


def pose_matrix(row):
    x, y, z, w = row[3:] / np.linalg.norm(row[3:])
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    return rotation @ FRD_FROM_CAMERA, row[:3]


def read_depth(path):
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None or encoded.ndim != 3 or encoded.shape[2] != 4 or encoded.dtype != np.uint8:
        raise ValueError(f"Expected four-channel float32 TartanAir depth PNG: {path}")
    return encoded.view("<f4")[..., 0]


def sample_depth(depth, pixels):
    rounded = np.rint(pixels).astype(int)
    y = np.clip(rounded[:, 1], 0, depth.shape[0] - 1)
    x = np.clip(rounded[:, 0], 0, depth.shape[1] - 1)
    return depth[y, x]


@dataclass
class Projection:
    focal: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    left_depth: np.ndarray
    right_depth: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

    def project(self, xy):
        """Return destination pixels and a depth consistency / visibility mask."""
        source = (np.asarray(xy) + 0.5) / self.scale - 0.5
        depth = sample_depth(self.left_depth, source)
        good_source = np.isfinite(depth) & (depth > 0.05) & (depth < 80)
        depth = np.where(good_source, depth, 1)
        points = np.column_stack(((source - self.center) / self.focal, np.ones(len(xy)))) * depth[:, None]
        moved = points @ self.rotation.T + self.translation
        z = moved[:, 2]
        target = (moved[:, :2] / np.where(z > 0.05, z, 1)[:, None]) * self.focal + self.center
        h, w = self.right_depth.shape
        in_frame = ((target[:, 0] >= 0) & (target[:, 0] < w - 1) &
                    (target[:, 1] >= 0) & (target[:, 1] < h - 1))
        destination_depth = sample_depth(self.right_depth, target)
        consistent = np.abs(destination_depth - z) <= np.maximum(0.05, z * 0.03)
        return (target + 0.5) * self.scale - 0.5, good_source & in_frame & (z > 0.05) & consistent

    def reversed(self):
        return Projection(self.focal, self.center, self.scale, self.right_depth, self.left_depth,
                          self.rotation.T, -self.rotation.T @ self.translation)

    def visible(self, xy):
        target, valid = self.project(xy)
        returned, reverse_valid = self.reversed().project(target)
        return target, valid & reverse_valid & (np.linalg.norm(returned - xy, axis=1) <= 1.5)


@dataclass
class FramePair:
    first: np.ndarray
    second: np.ndarray
    projection: Projection

    @cached_property
    def cells(self):
        height, width = self.first.shape
        x, y = np.meshgrid(np.arange(12, width - 4, 8), np.arange(12, height - 4, 8))
        points = np.column_stack((x.ravel(), y.ravel()))
        target, valid = self.projection.visible(points)
        target = target[valid].astype(int) // 8
        _, unique = np.unique(target, axis=0, return_index=True)
        return points[valid][unique] // 8, target[unique]


def load_pairs(manifest, width, height, splits):
    root = manifest.parent
    calibration = json.loads((root / "calibration.json").read_text())
    focal = np.array([calibration["fx"], calibration["fy"]])
    center = np.array([calibration["cx"], calibration["cy"]])
    scale = np.array([width / calibration["width"], height / calibration["height"]])
    native_shape = (calibration["height"], calibration["width"])
    pairs = {split: {} for split in splits}
    poses, verified = {}, {}
    with manifest.open(newline="") as stream:
        for row in csv.DictReader(stream):
            split = row["split"]
            if split not in pairs:
                continue
            directory = (root / row["group"]).resolve()
            if not directory.is_relative_to(root.resolve()):
                raise ValueError(f"Pair route is outside the dataset: {row['group']}")
            for key in ("rgb_a", "rgb_b", "depth_a", "depth_b", "pose"):
                path = (root / row[key]).resolve()
                if path.parent != directory:
                    raise ValueError(f"Pair member is outside the anchor route: {path}")
                if path not in verified:
                    verified[path] = sha256(path)
                if verified[path] != row[f"{key}_sha256"]:
                    raise ValueError(f"Geometric pair hash mismatch: {path}")
            pose_path = root / row["pose"]
            if pose_path not in poses:
                poses[pose_path] = np.loadtxt(pose_path, ndmin=2)
            trajectory = poses[pose_path]
            indexes = [int(row[f"frame_{key}"]) for key in ("a", "b")]
            if min(indexes) < 0 or max(indexes) >= len(trajectory):
                raise ValueError(f"Pose index outside trajectory: {pose_path}")
            pose_a, pose_b = trajectory[indexes]
            if not np.isfinite([pose_a, pose_b]).all() or min(np.linalg.norm(p[3:]) for p in (pose_a, pose_b)) == 0:
                raise ValueError(f"Invalid pose in {pose_path}")
            rotation_a, translation_a = pose_matrix(pose_a)
            rotation_b, translation_b = pose_matrix(pose_b)
            depth_a, depth_b = (read_depth(root / row[key]) for key in ("depth_a", "depth_b"))
            first, second = (cv2.imread(str(root / row[key]), cv2.IMREAD_GRAYSCALE) for key in ("rgb_a", "rgb_b"))
            if (first is None or second is None or
                    any(frame.shape != native_shape for frame in (depth_a, depth_b, first, second))):
                raise ValueError(f"Pair images do not match calibration: {row['rgb_a']}")
            projection = Projection(focal, center, scale, depth_a, depth_b,
                                    rotation_b.T @ rotation_a, rotation_b.T @ (translation_a - translation_b))
            pairs[split][(root / row["rgb_a"]).resolve()] = FramePair(
                cv2.resize(first, (width, height), interpolation=cv2.INTER_AREA),
                cv2.resize(second, (width, height), interpolation=cv2.INTER_AREA), projection)
    return pairs


def save_overlays(pairs, output):
    output.mkdir()
    for index, pair in enumerate(list(pairs.values())[:8]):
        h, w = pair.first.shape
        x, y = np.meshgrid(np.arange(12, w - 4, 28), np.arange(12, h - 4, 28))
        points = np.column_stack((x.ravel(), y.ravel()))
        canvas = cv2.cvtColor(np.hstack((pair.first, pair.second)), cv2.COLOR_GRAY2BGR)
        for direction, projection in enumerate((pair.projection, pair.projection.reversed())):
            target, valid = projection.visible(points)
            for start, end in zip(points[valid], np.rint(target[valid]).astype(int)):
                cv2.line(canvas, tuple(start + [w * direction, 0]),
                         tuple(end + [w * (1 - direction), 0]),
                         [(90, 215, 120), (255, 155, 70)][direction], 1, cv2.LINE_AA)
        if not cv2.imwrite(str(output / f"{index:02}.png"), canvas):
            raise OSError("Could not write geometric pair overlay")
