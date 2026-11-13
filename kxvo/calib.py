"""Checkerboard calibration of the real video path (camera, VTX, goggles, dongle).

Calibrate through the same HDMI capture chain and at the same resolution VO
will see, because the goggles and dongle may crop or scale the image.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .camera import Camera
from .video import VideoSource


def calibrate(src: str, cols: int, rows: int, square_m: float, width: int, fisheye: bool,
              uptilt_deg: float, want: int = 25, min_gap_s: float = 0.7, show: bool = True) -> Camera:
    video = VideoSource(src, width)
    objp = np.zeros((rows * cols, 1, 3), np.float64)
    objp[:, 0, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    obj_pts, img_pts = [], []
    last = 0.0
    size = None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    print(f"Move the {cols}x{rows} board around the whole frame. Collecting {want} views. q to stop.")
    while len(obj_pts) < want:
        t, gray, frame = video.read()
        if gray is None:
            break
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH)
        if found and time.monotonic() - last > min_gap_s:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit)
            obj_pts.append(objp.copy())
            img_pts.append(corners.reshape(-1, 1, 2).astype(np.float64))
            last = time.monotonic()
            print(f"  view {len(obj_pts)}/{want}")
        if show:
            vis = frame.copy()
            if found:
                cv2.drawChessboardCorners(vis, (cols, rows), corners, found)
            cv2.imshow("kxvo calibrate", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    video.close()
    if show:
        cv2.destroyAllWindows()
    if len(obj_pts) < 8:
        raise RuntimeError(f"only {len(obj_pts)} usable views; need at least 8")

    if fisheye:
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rms, K, D, _, _ = cv2.fisheye.calibrate(
            obj_pts, img_pts, size, K, D,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW, criteria=crit)
        dist = D.ravel().tolist()
        model = "fisheye"
    else:
        rms, K, D, _, _ = cv2.calibrateCamera([o.reshape(-1, 3).astype(np.float32) for o in obj_pts],
                                              [i.reshape(-1, 2).astype(np.float32) for i in img_pts],
                                              size, None, None)
        dist = D.ravel().tolist()[:5]
        model = "pinhole"
    print(f"RMS reprojection error {rms:.3f} px over {len(obj_pts)} views")
    return Camera(width=size[0], height=size[1], fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]),
                  cy=float(K[1, 2]), model=model, dist=dist, uptilt_deg=uptilt_deg)
