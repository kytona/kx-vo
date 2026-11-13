"""Synthetic benchmark: run the full pipeline on the simulated room and score it."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .geometry import rot_to_euler, wrap_pi
from .pipeline import Mount, Pipeline
from .synth import Room, default_camera, simulate
from .vo import VOConfig


@dataclass
class BenchResult:
    frames: int
    valid_ratio: float
    path_m: float
    rmse_m: float
    final_err_m: float
    final_err_pct: float
    max_err_m: float
    yaw_rmse_deg: float
    statuses: dict
    resets: int

    def summary(self) -> str:
        return (
            f"frames {self.frames}, valid {self.valid_ratio:.0%}, path {self.path_m:.1f} m, "
            f"RMSE {self.rmse_m:.3f} m, final {self.final_err_m:.3f} m ({self.final_err_pct:.1f}% of path), "
            f"max {self.max_err_m:.3f} m, yaw RMSE {self.yaw_rmse_deg:.2f} deg, resets {self.resets}, "
            f"statuses {self.statuses}"
        )


def run_synth(duration: float = 24.0, fps: float = 30.0, seed: int = 1, config: VOConfig | None = None,
              verbose: bool = False, **sim_kwargs) -> BenchResult:
    cam = default_camera()
    mount = Mount(camera_B=(0.03, 0.0, 0.0), rangefinder_B=(0.0, 0.0, 0.0))
    pipe = Pipeline(cam, mount, config)
    errs, yaw_errs, statuses = [], [], Counter()
    p0 = None
    prev_true = None
    path = 0.0
    last_err = 0.0
    n = 0
    for s in simulate(duration=duration, fps=fps, cam=cam, room=Room(), seed=seed,
                      cam_offset_B=mount.camera_B, **sim_kwargs):
        out = pipe.step(s.image, s.t, s.range_m, s.att_meas)
        n += 1
        statuses[out.status] += 1
        if p0 is None:
            p0 = s.p_WB.copy()
        true_rel = s.p_WB - p0
        if prev_true is not None:
            path += float(np.linalg.norm(true_rel[:2] - prev_true[:2]))
        prev_true = true_rel
        if out.valid:
            e = float(np.linalg.norm(out.p_WB[:2] - true_rel[:2]))
            errs.append(e)
            last_err = e
            yaw_true = rot_to_euler(s.R_WB)[2]
            yaw_errs.append(math.degrees(wrap_pi(out.rpy[2] - yaw_true)))
        if verbose and n % 30 == 0:
            print(f"t={s.t:5.1f} {out.status:9s} est={np.round(out.p_WB, 2)} true={np.round(true_rel, 2)} "
                  f"tracks={out.vo.n_tracks} lm={out.vo.n_landmarks} inl={out.vo.n_inliers} floor+={out.vo.n_floor} {out.vo.info}")
    errs_a = np.array(errs) if errs else np.array([np.inf])
    return BenchResult(
        frames=n,
        valid_ratio=len(errs) / max(n, 1),
        path_m=path,
        rmse_m=float(np.sqrt(np.mean(errs_a**2))),
        final_err_m=last_err,
        final_err_pct=100.0 * last_err / max(path, 1e-6),
        max_err_m=float(errs_a.max()),
        yaw_rmse_deg=float(np.sqrt(np.mean(np.array(yaw_errs) ** 2))) if yaw_errs else float("nan"),
        statuses=dict(statuses),
        resets=pipe.vo.reset_counter,
    )
