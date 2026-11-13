"""Frames and rotation helpers.

Frames used throughout kxvo:

* Camera (C): OpenCV convention, x right, y down, z forward (optical axis).
* Body (B): FRD, x forward, y right, z down.
* World (W): local NED (x north, y east, z down) fixed at VO start.

A rotation ``R_AB`` maps vectors from frame B into frame A: ``v_A = R_AB @ v_B``.
A pose ``(R_WX, p_WX)`` places frame X in the world.
"""

from __future__ import annotations

import math

import numpy as np

# Camera with zero uptilt: cam z -> body x, cam x -> body y, cam y -> body z.
R_BC_LEVEL = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX (yaw, pitch, roll) aerospace convention, body to world."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def rot_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`euler_to_rot`. Returns (roll, pitch, yaw) in radians."""
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def camera_to_body(uptilt_deg: float) -> np.ndarray:
    """R_BC for a forward camera pitched up by ``uptilt_deg``."""
    return rot_y(math.radians(uptilt_deg)) @ R_BC_LEVEL


def orthonormalize(M: np.ndarray) -> np.ndarray:
    """Closest rotation matrix to M (det +1)."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rotation_angle(R: np.ndarray) -> float:
    """Angle of a rotation matrix in radians."""
    c = (np.trace(R) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
