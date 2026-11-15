"""Run the pinned upstream MegaDepth-1500 protocol with explicit weights."""

import argparse
import importlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from .trackers import load_xfeat, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/xfeat-upstream"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Directory containing megadepth_test_1500")
    parser.add_argument("--seed", type=int, nargs="+", default=[1])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="Write the scores and checkpoint identity as JSON")
    args = parser.parse_args()
    if len(set(args.seed)) != len(args.seed):
        parser.error("Use distinct PoseLib seeds")
    model = load_xfeat(args.source, args.checkpoint, args.device)
    sys.path.insert(0, str(args.source.resolve()))
    try:
        upstream = importlib.import_module("modules.eval.megadepth1500")
    finally:
        sys.path.pop(0)
    # The pinned evaluator uses the NumPy 1.x integration name.
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
    np.random.seed(args.seed[0])
    cv2.setRNGSeed(args.seed[0])
    root = args.dataset / "megadepth_test_1500"
    if not root.is_dir():
        parser.error(f"Missing dataset directory: {root}")
    digest = sha256(args.checkpoint)
    print(f"checkpoint SHA-256 {digest}, seeds {args.seed}", flush=True)
    dataset = upstream.MegaDepth1500(json_file=str(args.source / "assets/megadepth_1500.json"), root_dir=str(root))
    loader = upstream.DataLoader(dataset, batch_size=1, shuffle=False)
    seeds = args.seed
    errors = {seed: [] for seed in seeds}
    for sample in upstream.tqdm.tqdm(loader):
        left, right = model.match_xfeat(upstream.tensor2bgr(sample["image0"]), upstream.tensor2bgr(sample["image1"]))
        left = left * sample["scale0"].numpy()
        right = right * sample["scale1"].numpy()
        K0, K1 = [sample[key].numpy()[0] for key in ("K0", "K1")]
        truth = sample["T_0to1"].numpy()[0]
        for seed in seeds:
            error = float("inf")
            if len(left) >= 5:
                pose, info = upstream.poselib.estimate_relative_pose(
                    left, right, upstream.intrinsics_to_camera(K0), upstream.intrinsics_to_camera(K1),
                    {"max_epipolar_error": 2.5, "success_prob": 0.99999,
                     "min_iterations": 20, "max_iterations": 1000, "seed": seed})
                if info.get("num_inliers", 0) >= 5:
                    error = max(upstream.relative_pose_error(truth, pose.R, pose.t))
            errors[seed].append(error)
    scores = {}
    for seed, values in errors.items():
        auc = {key: round(float(value) * 100, 3) for key, value in upstream.error_auc(values).items()}
        scores[seed] = auc
        print(f"PoseLib seed {seed}: {auc}")
    if args.output:
        result = {"checkpoint": str(args.checkpoint), "sha256": digest, "device": args.device,
                  "pairs": len(dataset), "max_epipolar_error_px": 2.5, "auc_percent": scores}
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
