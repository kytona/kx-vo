"""Glue between VO (camera frame) and the flight controller (body NED frame)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .camera import Camera
from .geometry import camera_to_body, euler_to_rot, rot_to_euler
from .vo import MonoVO, Observation, VOConfig, VOResult


@dataclass
class Mount:
    """Sensor positions on the airframe, body FRD, metres."""

    camera_B: tuple[float, float, float] = (0.03, 0.0, 0.0)
    rangefinder_B: tuple[float, float, float] = (0.0, 0.0, 0.01)


@dataclass
class PoseOut:
    t: float
    status: str
    valid: bool
    p_WB: np.ndarray
    rpy: tuple[float, float, float]
    reset_counter: int
    pos_sigma_m: float
    vo: VOResult
    extra: dict = field(default_factory=dict)


class Pipeline:
    """Runs MonoVO and reports the body pose in local NED.

    The flight controller's attitude is used twice: once to set the initial
    orientation (so VO's world frame is level and shares the FC's heading),
    and every frame to turn the slant rangefinder reading into vertical
    camera height. VO itself only uses the camera, so this is loose use of
    the FC, not tightly coupled visual-inertial odometry.
    """

    def __init__(self, camera: Camera, mount: Mount | None = None, config: VOConfig | None = None,
                 base_sigma_m: float = 0.05, tracker=None):
        self.cam = camera
        self.mount = mount or Mount()
        self.vo = MonoVO(camera, config, tracker)
        self.R_BC = camera_to_body(camera.uptilt_deg)
        self.base_sigma = base_sigma_m
        self.started = False

    def camera_height(self, range_m: float | None, R_WB: np.ndarray | None) -> float | None:
        if range_m is None or R_WB is None or not math.isfinite(range_m):
            return None
        down = R_WB @ np.array([0.0, 0.0, 1.0])  # body z axis in world
        sensor_h = range_m * down[2]  # vertical part of the slant range
        delta = R_WB @ (np.array(self.mount.camera_B) - np.array(self.mount.rangefinder_B))
        return float(sensor_h - delta[2])

    def step(self, img: np.ndarray, t: float, range_m: float | None,
             att: tuple[float, float, float] | None) -> PoseOut:
        R_WB_fc = euler_to_rot(*att) if att is not None else None
        if not self.started:
            R_WB0 = R_WB_fc if R_WB_fc is not None else np.eye(3)
            self.vo.reset(R_WB0 @ self.R_BC, R_WB0 @ np.array(self.mount.camera_B))
            self.started = True
        # Height uses the FC attitude when present, else VO's own attitude.
        R_for_h = R_WB_fc if R_WB_fc is not None else self.vo.R_WC @ self.R_BC.T
        obs = Observation(t=t, height_m=self.camera_height(range_m, R_for_h))
        res = self.vo.process(img, obs)
        R_WB = res.R_WC @ self.R_BC.T
        p_WB = res.p_WC - R_WB @ np.array(self.mount.camera_B)
        sigma = self.base_sigma * max(1.0, 20.0 / max(res.n_inliers, 1))
        return PoseOut(t, res.status, res.position_valid, p_WB, rot_to_euler(R_WB),
                       self.vo.reset_counter, sigma, res)
