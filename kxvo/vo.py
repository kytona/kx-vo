"""Camera-only monocular visual odometry with floor-anchored metric scale.

Monocular VO cannot observe scale by itself. Here scale comes from the floor:
the downward rangefinder gives camera height, so any tracked corner whose ray
points clearly below the horizon can be placed on the floor plane in metres.
Those floor landmarks, plus wall and furniture landmarks triangulated later
from metric poses, give a small local map. Every frame's pose is solved from
the map, with floor landmarks leading whenever there are enough of them, so
scale stays tied to measured height.

Per frame:

1. Track corners from the previous frame (pyramidal Lucas-Kanade with a
   forward-backward check), then re-track them from the last tracking
   keyframe so drift does not build up.
2. Solve the camera pose from tracks that have landmarks: guided, trimmed
   PnP from a constant-velocity prediction, then a refinement with camera
   height held to the rangefinder. Drop landmarks that stop reprojecting.
3. Create landmarks: floor points for corners below the horizon that pass a
   two-view floor check, triangulation for corners that have moved enough.
4. Detect new corners at keyframes when the track count drops.

Status values: ``ok`` (metric pose), ``no_scale`` (too few landmarks: hold,
publish nothing), ``lost`` (tracking collapsed: restart and bump the reset
counter), ``init``.

Known limits: no loop closure, so yaw and position drift grow with distance
(see README for synthetic numbers); the floor must be flat, textured and lit,
and the rangefinder must see the same floor the camera sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import Camera


@dataclass
class VOConfig:
    max_corners: int = 300
    quality: float = 0.01
    min_distance_px: int = 8
    replenish_below: int = 200
    min_tracks: int = 25
    fb_max_px: float = 1.0
    pnp_min_points: int = 12
    pnp_min_inliers: int = 10
    pnp_reproj_px: float = 2.0
    cull_reproj_px: float = 3.0
    floor_min_depression_deg: float = 15.0
    verify_baseline_m: float = 0.15  # motion before a bootstrap floor landmark is trusted
    verify_reproj_px: float = 1.5
    floor_check_baseline_m: float = 0.08  # second-view distance for the floor check
    floor_lead_points: int = 25  # use floor landmarks alone above this count
    floor_max_range_m: float = 6.0
    tri_min_angle_deg: float = 4.0
    kf_interval: int = 8  # frames between tracking keyframes (drift control)
    kf_refine_max_px: float = 2.0
    height_clamp: bool = True  # hold camera z to floor height from the rangefinder
    tri_max_depth_m: float = 12.0
    tri_reproj_px: float = 2.0
    max_step_m: float = 0.3  # largest believable motion between two frames


@dataclass
class Observation:
    """What VO knows about one frame beyond the image."""

    t: float
    height_m: float | None = None  # camera height above the floor, vertical


@dataclass
class VOResult:
    t: float
    status: str
    R_WC: np.ndarray
    p_WC: np.ndarray
    n_tracks: int = 0
    n_landmarks: int = 0
    n_inliers: int = 0
    n_floor: int = 0
    info: dict = field(default_factory=dict)

    @property
    def position_valid(self) -> bool:
        return self.status == "ok"


class MonoVO:
    def __init__(self, camera: Camera, config: VOConfig | None = None):
        self.cam = camera
        self.cfg = config or VOConfig()
        self.mask = camera.feature_mask()
        self.lk = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, R_WC: np.ndarray | None = None, p_WC: np.ndarray | None = None):
        self.R_WC = np.eye(3) if R_WC is None else np.array(R_WC, float)
        self.p_WC = np.zeros(3) if p_WC is None else np.array(p_WC, float)
        self.prev_img = None
        self.kf_img = None
        self.frames_since_kf = 0
        self.z_floor = None  # world z of the floor plane, set from first height
        self.vel = np.zeros(3)
        self.dt_last = 0.0
        self.t_last = None
        self.reset_counter = 0
        self._clear_tracks()

    def _clear_tracks(self):
        self.pts = np.empty((0, 2), np.float32)  # pixels, previous frame
        self.kf_pos = np.empty((0, 2), np.float32)  # pixels in the tracking keyframe
        self.lm = np.empty((0, 3))  # world landmark, NaN if none
        self.floor = np.empty(0, bool)  # landmark came from the floor plane
        self.verified = np.empty(0, bool)  # landmark confirmed by parallax
        self.birth_p = np.empty((0, 3))  # camera position when landmark was made
        self.ref_x = np.empty((0, 2))  # normalized coords at reference
        self.ref_R = np.empty((0, 3, 3))
        self.ref_p = np.empty((0, 3))

    def _keep(self, m: np.ndarray):
        self.pts, self.lm, self.floor = self.pts[m], self.lm[m], self.floor[m]
        self.kf_pos = self.kf_pos[m]
        self.verified, self.birth_p = self.verified[m], self.birth_p[m]
        self.ref_x, self.ref_R, self.ref_p = self.ref_x[m], self.ref_R[m], self.ref_p[m]

    def _detect(self, img: np.ndarray):
        mask = self.mask.copy()
        for x, y in self.pts.astype(int):
            cv2.circle(mask, (int(x), int(y)), self.cfg.min_distance_px, 0, -1)
        want = self.cfg.max_corners - len(self.pts)
        if want <= 0:
            return
        new = cv2.goodFeaturesToTrack(img, want, self.cfg.quality, self.cfg.min_distance_px, mask=mask)
        if new is None:
            return
        new = new.reshape(-1, 2).astype(np.float32)
        k = len(new)
        self.pts = np.vstack([self.pts, new])
        self.kf_pos = np.vstack([self.kf_pos, new])
        self.lm = np.vstack([self.lm, np.full((k, 3), np.nan)])
        self.floor = np.concatenate([self.floor, np.zeros(k, bool)])
        self.verified = np.concatenate([self.verified, np.zeros(k, bool)])
        self.birth_p = np.vstack([self.birth_p, np.repeat(self.p_WC[None], k, 0)])
        self.ref_x = np.vstack([self.ref_x, self.cam.normalize(new)])
        self.ref_R = np.concatenate([self.ref_R, np.repeat(self.R_WC[None], k, 0)])
        self.ref_p = np.vstack([self.ref_p, np.repeat(self.p_WC[None], k, 0)])

    def _track(self, img: np.ndarray):
        if not len(self.pts):
            return
        p0 = self.pts.reshape(-1, 1, 2)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev_img, img, p0, None, **self.lk)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(img, self.prev_img, p1, None, **self.lk)
        fb = np.linalg.norm((p0 - p0b).reshape(-1, 2), axis=1)
        h, w = img.shape[:2]
        p1 = p1.reshape(-1, 2)
        inside = (p1[:, 0] >= 2) & (p1[:, 0] < w - 2) & (p1[:, 1] >= 2) & (p1[:, 1] < h - 2)
        keep = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.cfg.fb_max_px) & inside
        # Re-track from the keyframe image, seeded with the frame-to-frame
        # result, so drift does not build up over long tracks.
        if self.kf_img is not None and len(self.kf_pos):
            guess = p1.reshape(-1, 1, 2).astype(np.float32).copy()
            p1k, stk, _ = cv2.calcOpticalFlowPyrLK(
                self.kf_img, img, self.kf_pos.reshape(-1, 1, 2), guess,
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk)
            p1k = p1k.reshape(-1, 2)
            agree = np.linalg.norm(p1k - p1, axis=1) < self.cfg.kf_refine_max_px
            keep &= (stk.ravel() == 1) & agree
            p1 = np.where(keep[:, None], p1k, p1)
        self.pts = p1.astype(np.float32)
        self._keep(keep)

    # ------------------------------------------------------------------- main
    def process(self, img: np.ndarray, obs: Observation) -> VOResult:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        info: dict = {}

        if self.prev_img is None:
            self.prev_img = img
            if obs.height_m is not None and math.isfinite(obs.height_m):
                self.z_floor = self.p_WC[2] + obs.height_m
            self._detect(img)
            self._new_keyframe(img)
            n_floor = self._add_floor_landmarks(obs.height_m, bootstrap=True)
            return VOResult(obs.t, "init", self.R_WC.copy(), self.p_WC.copy(), len(self.pts),
                            int(np.isfinite(self.lm[:, 0]).sum()), 0, n_floor)

        self.dt_last = 0.0 if self.t_last is None else max(0.0, obs.t - self.t_last)
        p_before = self.p_WC.copy()
        self._track(img)
        self.prev_img = img
        if len(self.pts) < self.cfg.min_tracks:
            self.reset_counter += 1
            self._clear_tracks()
            self._detect(img)
            self._new_keyframe(img)
            self._add_floor_landmarks(obs.height_m, bootstrap=True)
            return VOResult(obs.t, "lost", self.R_WC.copy(), self.p_WC.copy(), len(self.pts))

        status, n_inl = self._solve_pose(info, obs.height_m)
        if status == "ok" and self.dt_last > 0:
            self.vel = 0.7 * self.vel + 0.3 * (self.p_WC - p_before) / self.dt_last
        elif status != "ok":
            self.vel *= 0.5
        self.t_last = obs.t
        n_floor = 0
        if status == "ok":
            n_floor = self._add_floor_landmarks(obs.height_m)
            info["triangulated"] = self._triangulate()
        self.frames_since_kf += 1
        if self.frames_since_kf >= self.cfg.kf_interval or len(self.pts) < self.cfg.replenish_below:
            if len(self.pts) < self.cfg.replenish_below:
                self._detect(img)
                if status == "ok":
                    n_floor += self._add_floor_landmarks(obs.height_m)
            self._new_keyframe(img)
        if status != "ok" and int(np.isfinite(self.lm[:, 0]).sum()) < self.cfg.pnp_min_points:
            # Out of map: seed floor landmarks from the last pose to recover.
            n_floor += self._add_floor_landmarks(obs.height_m, bootstrap=True)
        n_lm = int(np.isfinite(self.lm[:, 0]).sum())
        return VOResult(obs.t, status, self.R_WC.copy(), self.p_WC.copy(), len(self.pts), n_lm, n_inl,
                        n_floor, info)

    # ---------------------------------------------------------------- helpers
    def _new_keyframe(self, img: np.ndarray):
        self.kf_img = img
        self.kf_pos = self.pts.copy()
        self.frames_since_kf = 0

    def _refine_fixed_z(self, R_CW: np.ndarray, p: np.ndarray, obj: np.ndarray, x: np.ndarray,
                        iters: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """Gauss-Newton over rotation and horizontal position with camera z held
        at the rangefinder height."""

        def resid(params):
            R = cv2.Rodrigues(params[:3].reshape(3, 1))[0] @ R_CW
            pp = np.array([p[0] + params[3], p[1] + params[4], p[2]])
            Xc = (obj - pp) @ R.T
            return (Xc[:, :2] / Xc[:, 2:3] - x).ravel()

        params = np.zeros(5)
        for _ in range(iters):
            r0 = resid(params)
            J = np.empty((len(r0), 5))
            for k in range(5):
                d = np.zeros(5)
                d[k] = 1e-6
                J[:, k] = (resid(params + d) - r0) / 1e-6
            step, *_ = np.linalg.lstsq(J, -r0, rcond=None)
            params += step
            if np.linalg.norm(step) < 1e-8:
                break
        R = cv2.Rodrigues(params[:3].reshape(3, 1))[0] @ R_CW
        return R, np.array([p[0] + params[3], p[1] + params[4], p[2]])

    def _solve_pose(self, info: dict, height_m: float | None = None) -> tuple[str, int]:
        has = np.isfinite(self.lm[:, 0])
        trusted = has & self.verified
        floor_ok = trusted & self.floor
        # Floor landmarks carry the rangefinder's scale, so they lead whenever
        # there are enough; triangulated ones fill in when the floor is scarce.
        if floor_ok.sum() >= self.cfg.floor_lead_points:
            has = floor_ok
        elif trusted.sum() >= 2 * self.cfg.pnp_min_points:
            has = trusted
        if has.sum() < self.cfg.pnp_min_points:
            info["reason"] = f"landmarks {int(has.sum())}"
            return "no_scale", 0
        obj = self.lm[has]
        x = self.cam.normalize(self.pts[has])
        # Centre the object points: with X' = X - m the translation is t' = t + R m.
        m = obj.mean(axis=0)
        obj_c = obj - m
        R_CW = self.R_WC.T
        p_pred = self.p_WC + self.vel * self.dt_last  # constant-velocity prediction
        rvec, _ = cv2.Rodrigues(R_CW)
        tvec_c = (-R_CW @ p_pred + R_CW @ m).reshape(3, 1)
        rvec, tvec_c, inl = self._robust_pnp(obj_c, x, rvec, tvec_c)
        n_inl = len(inl)
        if n_inl < self.cfg.pnp_min_inliers:
            info["reason"] = f"pnp inliers {n_inl}"
            return "no_scale", n_inl
        inl = inl.reshape(-1, 1)
        inl_idx = inl.ravel()
        rvec, tvec_c = cv2.solvePnPRefineLM(obj_c[inl_idx], x[inl_idx], np.eye(3), None, rvec, tvec_c)
        R_CW = cv2.Rodrigues(rvec)[0]
        tvec = tvec_c.ravel() - R_CW @ m
        p_new = -R_CW.T @ tvec
        if (self.cfg.height_clamp and self.z_floor is not None and height_m is not None
                and math.isfinite(height_m) and height_m > 0.05):
            p_fix = p_new.copy()
            p_fix[2] = self.z_floor - height_m
            R_CW, p_new = self._refine_fixed_z(R_CW, p_fix, obj[inl_idx], x[inl_idx])
            tvec = -R_CW @ p_new
        if np.linalg.norm(p_new - self.p_WC) > self.cfg.max_step_m:
            info["reason"] = "implausible jump"
            return "no_scale", n_inl
        self.R_WC, self.p_WC = R_CW.T, p_new
        info["culled"], info["verified"] = self._cull_and_verify(R_CW, tvec)
        return "ok", n_inl

    def _cull_and_verify(self, R_CW: np.ndarray, tvec: np.ndarray) -> tuple[int, int]:
        has = np.isfinite(self.lm[:, 0])
        if not has.any():
            return 0, 0
        idx = np.flatnonzero(has)
        x = self.cam.normalize(self.pts[idx])
        proj = self.lm[idx] @ R_CW.T + tvec.ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            err = np.linalg.norm(proj[:, :2] / proj[:, 2:3] - x, axis=1) * self.cam.focal
        bad = ~(err <= self.cfg.cull_reproj_px) | (proj[:, 2] <= 0)
        # Provisional floor landmarks must hold up under real motion.
        prov = ~self.verified[idx]
        moved = np.linalg.norm(self.p_WC - self.birth_p[idx], axis=1) > self.cfg.verify_baseline_m
        bad |= prov & moved & (err > self.cfg.verify_reproj_px)
        good_now = prov & moved & ~bad
        b = idx[bad]
        self.lm[b] = np.nan
        self.floor[b] = False
        self.verified[b] = False
        self.ref_x[b] = x[bad]
        self.ref_R[b] = self.R_WC
        self.ref_p[b] = self.p_WC
        self.verified[idx[good_now]] = True
        return int(bad.sum()), int(good_now.sum())

    def _robust_pnp(self, obj: np.ndarray, x: np.ndarray, rvec, tvec):
        """Guided trimmed PnP. OpenCV 5's RANSAC PnP is unreliable on near-planar
        floor sets, so start from the motion prediction, gate gross outliers,
        and tighten the inlier threshold over a few Levenberg-Marquardt solves.
        RANSAC without a guess is kept only as a fallback."""

        def errs(rv, tv):
            proj, _ = cv2.projectPoints(obj, rv, tv, np.eye(3), None)
            return np.linalg.norm(proj.reshape(-1, 2) - x, axis=1) * self.cam.focal

        best = (rvec, tvec, np.empty(0, int))
        rv, tv = rvec.copy(), tvec.copy()
        for gate in (25.0, 8.0, 4.0, self.cfg.pnp_reproj_px):
            sel = np.flatnonzero(errs(rv, tv) < gate)
            if len(sel) < 6:
                break
            ok, rv2, tv2 = cv2.solvePnP(obj[sel], x[sel], np.eye(3), None, rv.copy(), tv.copy(),
                                        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                break
            rv, tv = rv2, tv2
        inl = np.flatnonzero(errs(rv, tv) < self.cfg.pnp_reproj_px)
        if len(inl) >= len(best[2]):
            best = (rv, tv, inl)
        if len(best[2]) < max(self.cfg.pnp_min_inliers, int(0.4 * len(obj))):
            ok, rv3, tv3, inl3 = cv2.solvePnPRansac(obj, x, np.eye(3), None, iterationsCount=300,
                                                    reprojectionError=self.cfg.pnp_reproj_px / self.cam.focal,
                                                    confidence=0.999)
            if ok and inl3 is not None:
                inl3b = np.flatnonzero(errs(rv3, tv3) < self.cfg.pnp_reproj_px)
                if len(inl3b) > len(best[2]):
                    best = (rv3, tv3, inl3b)
        return best

    def _add_floor_landmarks(self, height_m: float | None, bootstrap: bool = False) -> int:
        """Place untracked corners that look below the horizon on the floor plane.

        Outside bootstrap, a candidate is accepted only if its floor point also
        reprojects onto where the corner was first seen, from a camera position
        far enough away to tell the floor from a wall (two-view check). At
        bootstrap there is no second view yet, so candidates enter provisional
        and must survive :meth:`_cull_and_verify`.
        """
        if height_m is None or not math.isfinite(height_m) or height_m < 0.05:
            return 0
        free = ~np.isfinite(self.lm[:, 0])
        if not free.any():
            return 0
        idx = np.flatnonzero(free)
        x = self.cam.normalize(self.pts[idx])
        rays = np.column_stack([x, np.ones(len(x))]) @ self.R_WC.T
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        dep = rays[:, 2] > math.sin(math.radians(self.cfg.floor_min_depression_deg))
        drop = (self.z_floor - self.p_WC[2]) if (self.cfg.height_clamp and self.z_floor is not None) else height_m
        s = drop / np.where(dep, rays[:, 2], np.inf)
        ok = dep & (s < self.cfg.floor_max_range_m)
        X = self.p_WC + rays * np.where(ok, s, 0.0)[:, None]
        if not bootstrap:
            base = np.linalg.norm(self.ref_p[idx] - self.p_WC, axis=1)
            Xr = np.einsum("nji,nj->ni", self.ref_R[idx], X - self.ref_p[idx])  # into reference camera
            with np.errstate(divide="ignore", invalid="ignore"):
                err = np.linalg.norm(Xr[:, :2] / Xr[:, 2:3] - self.ref_x[idx], axis=1) * self.cam.focal
            ok &= (base > self.cfg.floor_check_baseline_m) & (Xr[:, 2] > 0) & (err < self.cfg.verify_reproj_px)
        sel = idx[ok]
        self.lm[sel] = X[ok]
        self.floor[sel] = True
        self.verified[sel] = not bootstrap
        self.birth_p[sel] = self.p_WC
        return int(ok.sum())

    def _triangulate(self) -> int:
        free = ~np.isfinite(self.lm[:, 0])
        if not free.any():
            return 0
        idx = np.flatnonzero(free)
        x_cur = self.cam.normalize(self.pts[idx])
        d2 = np.column_stack([x_cur, np.ones(len(idx))]) @ self.R_WC.T
        d1 = np.einsum("nij,nj->ni", self.ref_R[idx], np.column_stack([self.ref_x[idx], np.ones(len(idx))]))
        d1 /= np.linalg.norm(d1, axis=1, keepdims=True)
        d2 /= np.linalg.norm(d2, axis=1, keepdims=True)
        cosang = np.clip(np.sum(d1 * d2, axis=1), -1, 1)
        enough = np.degrees(np.arccos(cosang)) > self.cfg.tri_min_angle_deg
        if not enough.any():
            return 0
        idx, d1, d2, x_cur = idx[enough], d1[enough], d2[enough], x_cur[enough]
        p1, p2 = self.ref_p[idx], np.repeat(self.p_WC[None], len(idx), 0)
        # Closest points between rays p1 + s1 d1 and p2 + s2 d2.
        w0 = p1 - p2
        b = np.sum(d1 * d2, 1)
        dd = np.sum(d1 * w0, 1)
        e = np.sum(d2 * w0, 1)
        den = 1 - b * b
        s1 = (b * e - dd) / den
        s2 = (e - b * dd) / den
        X = 0.5 * ((p1 + s1[:, None] * d1) + (p2 + s2[:, None] * d2))
        Xc = (X - self.p_WC) @ self.R_WC  # into current camera frame
        with np.errstate(divide="ignore", invalid="ignore"):
            reproj = np.linalg.norm(Xc[:, :2] / Xc[:, 2:3] - x_cur, axis=1) * self.cam.focal
        good = (s1 > 0.05) & (s2 > 0.05) & (Xc[:, 2] > 0.05) & (Xc[:, 2] < self.cfg.tri_max_depth_m) & (
            reproj < self.cfg.tri_reproj_px)
        self.lm[idx[good]] = X[good]
        self.verified[idx[good]] = True
        return int(good.sum())
