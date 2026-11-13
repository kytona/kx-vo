"""Run the pinned upstream MegaDepth-1500 protocol with explicit weights."""

import argparse
import importlib
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
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = load_xfeat(args.source, args.checkpoint, args.device)
    sys.path.insert(0, str(args.source.resolve()))
    try:
        upstream = importlib.import_module("modules.eval.megadepth1500")
    finally:
        sys.path.pop(0)
    # The pinned evaluator uses the NumPy 1.x integration name.
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
    np.random.seed(args.seed)
    cv2.setRNGSeed(args.seed)
    root = args.dataset / "megadepth_test_1500"
    if not root.is_dir():
        parser.error(f"Missing dataset directory: {root}")
    print(f"checkpoint SHA-256 {sha256(args.checkpoint)}, seed {args.seed}", flush=True)
    dataset = upstream.MegaDepth1500(json_file=str(args.source / "assets/megadepth_1500.json"), root_dir=str(root))
    loader = upstream.DataLoader(dataset, batch_size=1, shuffle=False)
    upstream.run_pose_benchmark(matcher_fn=model.match_xfeat, loader=loader, ransac_thr=2.5)


if __name__ == "__main__":
    main()
