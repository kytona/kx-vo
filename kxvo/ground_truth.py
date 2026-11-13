"""Extract an independent local NED reference from a calibrated overhead recording."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera


def extract(frame, detector, camera, world_from_camera, marker, marker_id, yaw_offset):
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None or marker_id not in ids:
        return None
    pixels = corners[np.flatnonzero(ids.ravel() == marker_id)[0]].reshape(4, 2)
    if camera.model == "fisheye":
        pixels = camera.normalize(pixels)
        K, dist = np.eye(3), None
    else:
        K, dist = camera.K, np.asarray(camera.dist) if camera.dist else None
    ok, rotation, translation = cv2.solvePnP(marker, pixels, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or translation[2, 0] <= 0:
        return None
    position = world_from_camera[:3, :3] @ translation.ravel() + world_from_camera[:3, 3]
    heading = world_from_camera[:3, :3] @ cv2.Rodrigues(rotation)[0][:, 0]
    return [*position, math.atan2(heading[1], heading[0]) + yaw_offset]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--camera", required=True, type=Path)
    parser.add_argument("--world-from-camera", required=True, type=Path,
                        help="Surveyed 4x4 rigid transform from OpenCV camera axes to local NED")
    parser.add_argument("--frame-times", required=True, type=Path,
                        help="One capture timestamp per frame, aligned to the VO log clock")
    parser.add_argument("--marker-id", required=True, type=int)
    parser.add_argument("--marker-m", required=True, type=float)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--yaw-offset-deg", type=float, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    camera = Camera.load(args.camera)
    matrix = np.loadtxt(args.world_from_camera)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1)):
        parser.error("Expected a proper 4x4 rigid transform")
    stamps = np.atleast_1d(np.loadtxt(args.frame_times))
    if stamps.ndim != 1 or not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
        parser.error("Frame times must be strictly increasing and finite")
    if args.marker_m <= 0:
        parser.error("Marker size must be positive")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    detector = cv2.aruco.ArucoDetector(dictionary)
    marker = args.marker_m / 2 * np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], np.float32)
    cap = cv2.VideoCapture(args.video)
    rows = []
    try:
        for t in stamps:
            ok, frame = cap.read()
            if not ok:
                parser.error("Fewer video frames than provided timestamps")
            if frame.shape[:2] != (camera.height, camera.width):
                parser.error("Video dimensions do not match overhead calibration")
            pose = extract(frame, detector, camera, matrix, marker, args.marker_id, math.radians(args.yaw_offset_deg))
            if pose is not None:
                rows.append([t, *pose])
        if cap.read()[0]:
            parser.error("More video frames than provided timestamps")
    finally:
        cap.release()
    with args.output.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["t", "x", "y", "z", "yaw_rad"])
        writer.writerows(rows)
    print(f"Tracked {len(rows)} of {len(stamps)} overhead frames")


if __name__ == "__main__":
    main()
