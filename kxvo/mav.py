"""MAVLink link to the flight controller: telemetry in, VO pose out.

Output is ``VISION_POSITION_ESTIMATE`` in local NED. ArduPilot consumes it as
external navigation (``VISO_TYPE`` = MAVLink), and the KX Betaflight fork's
P3 patch fuses the same message into its position estimator.

kxvo only publishes estimates. It never sends arming, mode or setpoint
commands, and it stops publishing whenever VO is not confident, so the flight
controller falls back to its own sources on its own timeout.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass

# MAVLink 2 is required: covariance and reset_counter are extension fields.
os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil  # noqa: E402

MAV_COMP_ID_VISUAL_INERTIAL_ODOMETRY = 197


@dataclass
class Telemetry:
    att: tuple[float, float, float] | None = None
    att_t: float = 0.0
    range_m: float | None = None
    range_t: float = 0.0
    heartbeat_t: float = 0.0
    autopilot: str = "unknown"


class MavLink:
    def __init__(self, url: str, source_system: int = 1, rate_hz: float = 30.0):
        self.conn = mavutil.mavlink_connection(
            url, source_system=source_system, source_component=MAV_COMP_ID_VISUAL_INERTIAL_ODOMETRY,
            dialect="ardupilotmega", autoreconnect=True,
        )
        self.tel = Telemetry()
        self.rate_hz = rate_hz
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._rx = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx.start()
        self._last_hb = 0.0
        self._requested = False

    # --------------------------------------------------------------- receive
    def _rx_loop(self):
        while not self._stop.is_set():
            try:
                msg = self.conn.recv_match(blocking=True, timeout=0.2)
            except OSError:
                if self._stop.is_set():
                    return
                time.sleep(0.1)
                continue
            if msg is None:
                continue
            now = time.monotonic()
            kind = msg.get_type()
            with self._lock:
                if kind == "HEARTBEAT" and msg.get_srcComponent() == 1:
                    self.tel.heartbeat_t = now
                    self.tel.autopilot = {3: "ardupilot", 12: "px4"}.get(msg.autopilot, str(msg.autopilot))
                elif kind == "ATTITUDE":
                    self.tel.att = (msg.roll, msg.pitch, msg.yaw)
                    self.tel.att_t = now
                elif kind == "DISTANCE_SENSOR" and msg.orientation == 25:  # MAV_SENSOR_ROTATION_PITCH_270
                    self.tel.range_m = msg.current_distance / 100.0
                    self.tel.range_t = now
                elif kind == "RANGEFINDER":
                    self.tel.range_m = float(msg.distance)
                    self.tel.range_t = now

    def snapshot(self, max_age_s: float = 0.2) -> tuple[tuple | None, float | None]:
        """Latest attitude and range, or None for anything older than max_age_s."""
        att, rng, _ = self.snapshot_with_ages(max_age_s)
        return att, rng

    def snapshot_with_ages(self, max_age_s=0.2):
        with self._lock:
            now = time.monotonic()
            att_age = now - self.tel.att_t if self.tel.att_t else None
            range_age = now - self.tel.range_t if self.tel.range_t else None
            att = self.tel.att if att_age is not None and att_age < max_age_s else None
            rng = self.tel.range_m if range_age is not None and range_age < max_age_s else None
        return att, rng, {"attitude": att_age, "range": range_age}

    def request_streams(self):
        """Ask the FC for attitude and rangefinder at a useful rate."""
        interval_us = int(1e6 / self.rate_hz)
        for msg_id in (30, 132):  # ATTITUDE, DISTANCE_SENSOR
            self.conn.mav.command_long_send(
                self.conn.target_system or 1, self.conn.target_component or 1,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, msg_id, interval_us, 0, 0, 0, 0, 0)
        self._requested = True

    # ------------------------------------------------------------------ send
    def heartbeat(self):
        now = time.monotonic()
        if now - self._last_hb >= 1.0:
            self.conn.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                                         mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            self._last_hb = now
            if not self._requested and self.tel.heartbeat_t:
                self.request_streams()

    def send_pose(self, t_capture: float, p_ned, rpy, sigma_m: float, reset_counter: int, compact: bool = True):
        self.conn.mav.send(vision_position_estimate(t_capture, p_ned, rpy, sigma_m, reset_counter, self.conn.mav,
                                                    compact=compact))

    def close(self):
        self._stop.set()
        self._rx.join(timeout=1.0)
        self.conn.close()


def covariance(sigma_m: float, sigma_rad: float = math.radians(2.0)) -> list[float]:
    """Row-major upper triangle of the 6x6 pose covariance (x, y, z, roll, pitch, yaw)."""
    cov = [0.0] * 21
    diag_idx = [0, 6, 11, 15, 18, 20]
    var = [sigma_m**2] * 3 + [sigma_rad**2] * 3
    for i, v in zip(diag_idx, var):
        cov[i] = v
    return cov


def vision_position_estimate(t_capture: float, p_ned, rpy, sigma_m: float, reset_counter: int, mav,
                             compact: bool = False):
    """Encode VISION_POSITION_ESTIMATE.

    ``compact`` sends an all-zero covariance, which MAVLink 2 trims off the
    wire (about 45 bytes per message instead of about 130). The flight
    controller then uses its own noise setting (ArduPilot ``VISO_POS_M_NSE``,
    KX Betaflight ``kx_extnav_pos_noise``). Use it on ELRS links.
    """
    usec = int(t_capture * 1e6)
    cov = [0.0] * 21 if compact else covariance(sigma_m)
    return mav.vision_position_estimate_encode(
        usec, float(p_ned[0]), float(p_ned[1]), float(p_ned[2]),
        float(rpy[0]), float(rpy[1]), float(rpy[2]), cov, int(reset_counter) & 0xFF)
