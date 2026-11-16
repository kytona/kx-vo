"""Compare frozen XFeat checkpoints on manifest test frames and known warps."""

import argparse
import json
from pathlib import Path

from .trackers import load_xfeat, sha256
from .train import evaluate_pairs, load_images, read_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--xfeat-source", type=Path, default=Path("runs/xfeat-upstream"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--max-blur", type=int, default=11)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.width, args.height) < 32 or args.width % 32 or args.height % 32 or args.max_blur < 3:
        parser.error("Use dimensions divisible by 32 and max blur >= 3")
    rows = read_manifest(args.manifest, required_splits=("test",))
    images = load_images(rows, ("test",), args.width, args.height)["test"]
    models = []
    for checkpoint in args.checkpoint:
        model = load_xfeat(args.xfeat_source, checkpoint, args.device)
        scores = evaluate_pairs(model, images, args.max_blur, seed=args.seed)
        print(f"{checkpoint}: clean recall {scores['clean']['recall']:.3f}, "
              f"blur recall {scores['blur']['recall']:.3f}", flush=True)
        models.append({"checkpoint": str(checkpoint), "sha256": sha256(checkpoint), **scores})
    result = {"manifest_sha256": sha256(args.manifest), "frames": len(images),
              "groups": sorted({row["group"] for row in rows if row["split"] == "test"}),
              "pair_seed": args.seed, "size": [args.width, args.height], "max_blur": args.max_blur,
              "pixel_tolerance": 3, "device": args.device, "models": models}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
