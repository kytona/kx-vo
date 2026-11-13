"""Camera intrinsics, distortion and calibration file I/O."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Camera:
    """Intrinsics for the image size VO actually processes.

    ``model`` is ``"pinhole"`` (OpenCV k1, k2, p1, p2, k3) or ``"fisheye"``
    (OpenCV fisheye k1..k4). ``uptilt_deg`` is the camera pitch-up angle on
    the airframe. ``mask`` holds pixel rectangles ``[x, y, w, h]`` that VO
    must ignore, for example goggle OSD elements burned into the HDMI feed.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "pinhole"
    dist: list[float] = field(default_factory=list)
    uptilt_deg: float = 0.0
    mask: list[list[int]] = field(default_factory=list)

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    @property
    def focal(self) -> float:
        return 0.5 * (self.fx + self.fy)

    def normalize(self, pts: np.ndarray) -> np.ndarray:
        """Pixel points (N, 2) to undistorted normalized image coordinates."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.empty((0, 2))
        if self.model == "fisheye":
            d = np.array((self.dist + [0.0] * 4)[:4], dtype=np.float64)
            out = cv2.fisheye.undistortPoints(pts, self.K, d)
        else:
            d = np.array(self.dist, dtype=np.float64) if self.dist else None
            out = cv2.undistortPoints(pts, self.K, d)
        return out.reshape(-1, 2)

    def scaled(self, width: int) -> "Camera":
        """Same camera at a different processing width (aspect kept)."""
        s = width / self.width
        return Camera(
            width=width,
            height=int(round(self.height * s)),
            fx=self.fx * s,
            fy=self.fy * s,
            cx=self.cx * s,
            cy=self.cy * s,
            model=self.model,
            dist=list(self.dist),
            uptilt_deg=self.uptilt_deg,
            mask=[[int(round(v * s)) for v in r] for r in self.mask],
        )

    def feature_mask(self) -> np.ndarray:
        """uint8 mask, 255 where features may be detected."""
        m = np.full((self.height, self.width), 255, np.uint8)
        border = 4
        m[:border, :] = 0
        m[-border:, :] = 0
        m[:, :border] = 0
        m[:, -border:] = 0
        for x, y, w, h in self.mask:
            m[max(0, y) : y + h, max(0, x) : x + w] = 0
        return m

    @classmethod
    def load(cls, path: str | Path) -> "Camera":
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2) + "\n")
