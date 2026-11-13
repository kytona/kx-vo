"""Synthetic indoor room for testing VO without hardware.

A ray-cast textured box (floor, four walls, ceiling) rendered through a
pinhole camera mounted on a simulated FlyTimes-sized whoop. It produces
frames, the true pose, and simulated rangefinder and attitude readings with
noise, so the whole pipeline can be measured against ground truth.

This tests geometry and plumbing only. It has no motion blur, rolling
shutter, lighting change, compression or real latency, so synthetic drift
numbers are a best case and are not flight evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Camera
from .geometry import camera_to_body, euler_to_rot


@dataclass
class Room:
    size_x: float = 8.0  # north, m
    size_y: float = 6.0  # east, m
    height: float = 3.0  # floor to ceiling, m
    texel_m: float = 0.004
    seed: int = 7

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        lo = np.array([0.0, 0.0, -self.height])
        hi = np.array([self.size_x, self.size_y, 0.0])
        self.lo, self.hi = lo, hi
        # face key: (axis, side) -> texture spanning the two other axes
        self.tex = {}
        for axis in range(3):
            others = [a for a in range(3) if a != axis]
            span = [hi[a] - lo[a] for a in others]
            shape = (int(span[1] / self.texel_m) + 2, int(span[0] / self.texel_m) + 2)
            for side in (0, 1):
                self.tex[(axis, side)] = _texture(shape, rng, floor=(axis == 2 and side == 1))


def _texture(shape: tuple[int, int], rng: np.random.Generator, floor: bool) -> np.ndarray:
    h, w = shape
    img = np.zeros((h, w), np.float32)
    for cells, amp in ((6, 60.0), (30, 45.0), (120, 35.0), (500, 20.0)):
        small = rng.random((max(2, int(cells * h / max(h, w))), max(2, cells))).astype(np.float32)
        img += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    # Hard-edged blobs give corners, like posters, tiles and furniture.
    n_blobs = int(w * h / (60 * 60)) if floor else int(w * h / (90 * 90))
    for _ in range(n_blobs):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(6, 30))
        val = float(rng.uniform(-70, 70))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x - r, y - r), (x + r, y + r // 2), val, -1)
        else:
            cv2.circle(img, (x, y), r, val, -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    img -= img.min()
    img *= 230.0 / max(img.max(), 1e-6)
    return (img + 12).astype(np.float32)


def render(room: Room, cam: Camera, R_WC: np.ndarray, p_WC: np.ndarray, noise: float = 2.0,
           rng: np.random.Generator | None = None) -> np.ndarray:
    """Render a uint8 grayscale frame of ``room`` from camera pose (R_WC, p_WC)."""
    u, v = np.meshgrid(np.arange(cam.width, dtype=float), np.arange(cam.height, dtype=float))
    rays_c = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u)], -1)
    d = rays_c.reshape(-1, 3) @ R_WC.T
    o = p_WC
    with np.errstate(divide="ignore", invalid="ignore"):
        t_hi = (room.hi - o) / d
        t_lo = (room.lo - o) / d
    t_exit = np.where(d > 0, t_hi, t_lo)
    t_exit = np.where(np.isfinite(t_exit) & (t_exit > 0), t_exit, np.inf)
    axis = np.argmin(t_exit, axis=1)
    t = t_exit[np.arange(len(axis)), axis]
    hit = o + d * t[:, None]
    side = (d[np.arange(len(axis)), axis] > 0).astype(int)
    out = np.zeros(len(axis), np.float32)
    for (ax, sd), tex in room.tex.items():
        sel = (axis == ax) & (side == sd)
        if not sel.any():
            continue
        others = [a for a in range(3) if a != ax]
        mx = ((hit[sel, others[0]] - room.lo[others[0]]) / room.texel_m).astype(np.float32)
        my = ((hit[sel, others[1]] - room.lo[others[1]]) / room.texel_m).astype(np.float32)
        k = len(mx)
        cols = 256  # cv2.remap maps must stay below 32767 per dimension
        pad = (-k) % cols
        mx = np.pad(mx, (0, pad)).reshape(-1, cols)
        my = np.pad(my, (0, pad)).reshape(-1, cols)
        vals = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out[sel] = vals.ravel()[:k]
    img = out.reshape(cam.height, cam.width)
    if noise > 0:
        rng = rng or np.random.default_rng()
        img = img + rng.normal(0.0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


@dataclass
class SimSample:
    t: float
    image: np.ndarray
    R_WB: np.ndarray
    p_WB: np.ndarray
    R_WC: np.ndarray
    p_WC: np.ndarray
    range_m: float  # downward rangefinder reading (slant)
    att_meas: tuple[float, float, float]  # noisy roll, pitch, yaw


def default_camera() -> Camera:
    return Camera(width=320, height=240, fx=150.0, fy=150.0, cx=160.0, cy=120.0, uptilt_deg=15.0)


def loop_trajectory(t: float, center=(4.0, 3.0), radius=(1.6, 1.1), period=24.0, alt=1.0):
    """Smooth closed loop at constant height with heading along the path and a
    little roll/pitch wobble, like a gentle indoor whoop flight."""
    w = 2 * math.pi / period
    x = center[0] + radius[0] * math.sin(w * t)
    y = center[1] + radius[1] * math.sin(2 * w * t) / 2 + radius[1] * 0.5 * math.sin(w * t)
    z = -alt - 0.15 * math.sin(0.7 * w * t)
    dx = radius[0] * w * math.cos(w * t)
    dy = radius[1] * w * math.cos(2 * w * t) + radius[1] * 0.5 * w * math.cos(w * t)
    yaw = math.atan2(dy, dx)
    roll = math.radians(4.0) * math.sin(1.3 * w * t)
    pitch = math.radians(-6.0) + math.radians(3.0) * math.sin(1.7 * w * t)
    return np.array([x, y, z]), (roll, pitch, yaw)


def simulate(duration: float = 24.0, fps: float = 30.0, cam: Camera | None = None, room: Room | None = None,
             range_sigma: float = 0.01, att_sigma_deg: float = 0.3, yaw_drift_dps: float = 0.2,
             seed: int = 1, trajectory=loop_trajectory, cam_offset_B=(0.03, 0.0, 0.0)):
    """Yield SimSample frames along ``trajectory``."""
    cam = cam or default_camera()
    room = room or Room()
    rng = np.random.default_rng(seed)
    R_BC = camera_to_body(cam.uptilt_deg)
    off = np.array(cam_offset_B)
    n = int(duration * fps)
    for i in range(n):
        t = i / fps
        p_WB, (roll, pitch, yaw) = trajectory(t)
        R_WB = euler_to_rot(roll, pitch, yaw)
        R_WC = R_WB @ R_BC
        p_WC = p_WB + R_WB @ off
        img = render(room, cam, R_WC, p_WC, rng=rng)
        height = -p_WB[2]
        slant = height / max(1e-3, math.cos(roll) * math.cos(pitch))
        s = math.radians(att_sigma_deg)
        att = (roll + rng.normal(0, s), pitch + rng.normal(0, s),
               yaw + rng.normal(0, s) + math.radians(yaw_drift_dps) * t)
        yield SimSample(t, img, R_WB, p_WB, R_WC, p_WC, slant + rng.normal(0, range_sigma), att)
