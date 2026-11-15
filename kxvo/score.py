"""Score replay tracking health and, when supplied, error against ground truth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def load_reference(path):
    if path.suffix == ".csv":
        with path.open(newline="") as stream:
            return np.array([[float(row[k]) for k in ("t", "x", "y", "z", "yaw_rad")]
                             for row in csv.DictReader(stream)])
    return np.array([[r["t"], *r["truth_p"], r["truth_rpy"][2]]
                     for r in map(json.loads, path.read_text().splitlines()) if "truth_p" in r])


def tracking_health(records):
    t = np.array([row["t"] for row in records])
    if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Need at least two finite, strictly increasing replay timestamps")
    valid = np.array([bool(row["valid"]) for row in records])
    valid_indices = np.flatnonzero(valid)
    # Timestamp gaps describe coverage only between the first and last logged samples.
    durations = np.diff(t)
    sums = np.r_[0, np.cumsum(durations[~valid[:-1]])]
    intervals = np.flatnonzero(~valid[:-1])
    boundaries = np.r_[0, np.flatnonzero(np.diff(intervals) > 1) + 1, len(intervals)]
    outages = np.diff(sums[boundaries])
    metrics = {"frames": len(t), "valid_frames": int(valid.sum()),
               "valid_fraction": float(valid.mean()), "duration_s": float(t[-1] - t[0]),
               "initialization_s": float(t[valid_indices[0]] - t[0]) if len(valid_indices) else None,
               "unavailable_s": float(sums[-1]),
               "longest_unavailable_s": float(outages.max(initial=0)),
               "reset_changes": sum(a.get("reset", 0) != b.get("reset", 0)
                                    for a, b in zip(records, records[1:]))}
    ms = [row["processing_ms"] for row in records if row.get("processing_ms") is not None]
    if ms:
        metrics["processing_ms_p50_p95"] = np.percentile(ms, [50, 95]).tolist()
    return metrics


def score(records, ref, alignment_seconds=1.0, window=5.0, max_ref_gap=0.15, alignment_start=None):
    t = np.array([r["t"] for r in records])
    if len(t) < 2 or len(ref) < 2 or not np.isfinite(ref).all() or np.any(np.diff(t) <= 0) or np.any(np.diff(ref[:, 0]) <= 0):
        raise ValueError("Need strictly increasing estimate and reference timestamps with finite ground truth")
    indices = np.searchsorted(ref[:, 0], t, side="right")
    lo, hi = np.maximum(indices - 1, 0), np.minimum(indices, len(ref) - 1)
    covered = ((t >= ref[0, 0]) & (t <= ref[-1, 0])
               & (ref[hi, 0] - ref[lo, 0] <= max_ref_gap))
    valid = covered & np.array([bool(r["valid"]) for r in records])
    truth = np.column_stack([np.interp(t, ref[:, 0], ref[:, i]) for i in (1, 2)])
    truth_yaw = np.interp(t, ref[:, 0], np.unwrap(ref[:, 4]))
    estimate = np.array([r["p"][:2] for r in records])
    yaw = np.array([r["rpy"][2] for r in records])
    accepted = np.flatnonzero(valid)
    result = {"frames": len(t), "reference_covered_frames": int(covered.sum()),
              "valid_covered_frames": int(valid.sum()), "reset_changes": sum(
                  a.get("reset", 0) != b.get("reset", 0) for a, b in zip(records, records[1:])),
              "initialization_s": float(t[accepted[0]] - t[0]) if len(accepted) else None}
    ms = [r["processing_ms"] for r in records if r.get("processing_ms") is not None]
    if ms:
        result["processing_ms_p50_p95"] = np.percentile(ms, [50, 95]).tolist()
    if not len(accepted):
        return result, None
    start = t[accepted[0]]
    # Count tracker outages from estimator status independently of reference gaps.
    healthy = np.flatnonzero([r["valid"] for r in records])
    gaps = np.diff(np.r_[t[healthy], t[-1]])
    outages = gaps[gaps > 0.2]
    result.update(outage_count=len(outages), outage_s=float(outages.sum()),
                  unavailable_s=float(np.diff(t)[~np.array([r["valid"] for r in records[:-1]], bool)].sum()))
    fit_start = ref[0, 0] + 1 if alignment_start is None else alignment_start
    fit = valid & (t >= fit_start) & (t <= fit_start + alignment_seconds)
    use = valid & (t > fit_start + alignment_seconds)
    result["alignment_window_s"] = [float(fit_start), float(fit_start + alignment_seconds)]
    if fit.sum() < 3 or not use.any():
        return result, None
    offset = np.angle(np.mean(np.exp(1j * (truth_yaw[fit] - yaw[fit]))))
    c, s = np.cos(offset), np.sin(offset)
    rotate = np.array([[c, s], [-s, c]])
    aligned = estimate @ rotate
    aligned += np.mean(truth[fit] - aligned[fit], axis=0)
    yaw_error = wrap(yaw + offset - truth_yaw)
    error = np.linalg.norm(aligned - truth, axis=1)
    eligible = np.flatnonzero(use)
    end = eligible[-1]
    path_m = float(np.linalg.norm(np.diff(truth[covered], axis=0), axis=1).sum())
    result.update(horizontal_rms_m=float(np.sqrt(np.mean(error[use] ** 2))),
                  yaw_rms_deg=float(np.degrees(np.sqrt(np.mean(yaw_error[use] ** 2)))),
                  endpoint_error_m=float(error[end]), path_m=path_m,
                  endpoint_error_pct=float(100 * error[end] / path_m) if path_m > 0 else None)
    partners = np.searchsorted(t, t[eligible] + window)
    i = eligible[partners < len(t)]
    j = partners[partners < len(t)]
    resets = np.array([r.get("reset", 0) for r in records])
    pairs = use[j] & (resets[i] == resets[j]) & (t[j] - t[i] <= window + 0.1)
    i, j = i[pairs], j[pairs]
    result["relative_pairs"] = len(i)
    if len(i):
        delta = (aligned[j] - aligned[i]) - (truth[j] - truth[i])
        result["relative_translation_m_median"] = float(np.median(np.linalg.norm(delta, axis=1)))
        result["relative_yaw_deg_median"] = float(np.degrees(np.median(np.abs(wrap(yaw_error[j] - yaw_error[i])))))
    return result, (truth[covered], aligned[valid])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, help="Time-aligned pose reference for trajectory error")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--alignment-seconds", type=float, default=1)
    parser.add_argument("--alignment-start", type=float, help="Start of the fixed clean window in the reference clock")
    parser.add_argument("replays", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if args.plot and args.reference is None:
        parser.error("--plot requires --reference")
    reference = load_reference(args.reference) if args.reference else None
    results = {}
    if args.plot:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
        ax.plot(reference[:, 2], reference[:, 1], color="black", label="reference")
    for path in args.replays:
        records = [r for r in map(json.loads, path.read_text().splitlines()) if "p" in r]
        if reference is None:
            results[str(path)] = tracking_health(records)
            continue
        metrics, trajectory = score(records, reference, alignment_seconds=args.alignment_seconds,
                                    alignment_start=args.alignment_start)
        results[str(path)] = metrics
        if args.plot and trajectory is not None:
            ax.plot(trajectory[1][:, 1], trajectory[1][:, 0], label=path.stem)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    if args.plot:
        ax.set(xlabel="East (m)", ylabel="North (m)", aspect="equal")
        ax.legend()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.plot, bbox_inches="tight")


if __name__ == "__main__":
    main()
