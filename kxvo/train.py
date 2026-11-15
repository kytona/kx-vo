"""Warm-start sparse XFeat with deterministic supervised image pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .trackers import XFEAT_REV, load_xfeat, sha256
from .video import processing_frame

OFFICIAL_SHA = "0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b"


def project(points, warp):
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2), warp).reshape(-1, 2)


def inside(points, width, height, border=4):
    return ((points[:, 0] >= border) & (points[:, 0] < width - border)
            & (points[:, 1] >= border) & (points[:, 1] < height - border))


def motion_blur(image, length, angle):
    size = max(1, int(length) | 1)
    center = (size - 1) / 2
    delta = center * np.array([math.cos(angle), math.sin(angle)])
    kernel = np.zeros((size, size), np.float32)
    cv2.line(kernel, tuple(np.rint(center - delta).astype(int)),
             tuple(np.rint(center + delta).astype(int)), 1, 1)
    return cv2.filter2D(image, -1, kernel / kernel.sum())


def pair(image, rng, blur, max_blur, blur_both=False):
    h, w = image.shape
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    shift = rng.uniform(-0.06, 0.06, (4, 2)) * [w, h]
    warp = cv2.getPerspectiveTransform(corners, (corners + shift).astype(np.float32))
    second = cv2.warpPerspective(image, warp, (w, h))
    if blur:
        second = motion_blur(second, rng.integers(3, max_blur + 1), rng.uniform(0, math.pi))
        if blur_both:
            image = motion_blur(image, rng.integers(3, max_blur + 1), rng.uniform(0, math.pi))
    return image, second, warp


def labelled_cells(warp, w, h):
    # Interior cell centers give one positive correspondence per descriptor cell.
    x, y = np.meshgrid(np.arange(12, w - 4, 8), np.arange(12, h - 4, 8))
    a = np.column_stack((x.ravel(), y.ravel()))
    b = project(a, warp)
    valid = inside(b, w, h)
    a, b = a[valid] // 8, b[valid].astype(int) // 8
    _, unique = np.unique(b, axis=0, return_index=True)
    return a[unique], b[unique]


def image_tensor(image, device):
    import torch
    return torch.as_tensor(image.copy(), dtype=torch.float32, device=device)[None, None] / 255


def descriptor_loss(first, second, a, b):
    import torch
    from torch.nn import functional as F
    # Bidirectional correspondence classification as in upstream XFeat.
    x = first[:, a[:, 1], a[:, 0]].T
    y = second[:, b[:, 1], b[:, 0]].T
    logits = (x @ y.T) * 0.2
    target = torch.arange(len(a), device=logits.device)
    return F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)


def read_manifest(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups, hashes = {}, {}
    for row in rows:
        row["file"] = path.parent / row["path"]
        split, group = row["split"], row["group"]
        if split not in {"train", "validation", "test"} or not group or not row.get("rights"):
            raise ValueError("Every row needs a valid split, group and rights record")
        if groups.setdefault(group, split) != split:
            raise ValueError(f"Group crosses splits: {group}")
        digest = sha256(row["file"])
        if row.get("sha256") and row["sha256"] != digest:
            raise ValueError(f"Image hash does not match manifest: {row['path']}")
        row["sha256"] = digest
        if hashes.setdefault(row["sha256"], split) != split:
            raise ValueError(f"Identical image crosses splits: {row['path']}")
    for split in ("train", "validation"):
        if not any(row["split"] == split for row in rows):
            raise ValueError(f"No {split} images")
    return rows


def evaluate_pairs(model, images, max_blur, seed=314):
    import torch
    result = {}
    model.eval()
    for name, degraded in (("clean", False), ("blur", True)):
        rng = np.random.default_rng(seed)
        correct = total = visible = 0
        with torch.inference_mode():
            for image in images:
                a, b, warp = pair(image, rng, degraded, max_blur)
                left = model.detectAndCompute(a, top_k=400)[0]
                right = model.detectAndCompute(b, top_k=400)[0]
                xy = left["keypoints"].cpu().numpy()
                target = project(xy, warp) if len(xy) else xy
                valid = inside(target, image.shape[1], image.shape[0])
                visible += int(valid.sum())
                if len(xy) and len(right["keypoints"]):
                    m, n = model.match(left["descriptors"], right["descriptors"], 0.75)
                    m = m.cpu().numpy()
                    n = n.cpu().numpy()
                    dist = np.linalg.norm(right["keypoints"].cpu().numpy()[n] - target[m], axis=1)
                    total += int(valid[m].sum())
                    correct += int((valid[m] & (dist <= 3)).sum())
        result[name] = {"precision": correct / max(total, 1), "recall": correct / max(visible, 1),
                        "correct": correct, "matches": total, "visible": visible}
    return result


def save_validation_overlays(images, max_blur, output):
    """Show known correspondences and augmentations for manual label inspection."""
    directory = output / "pair-overlays"
    directory.mkdir()
    rng = np.random.default_rng(314)
    for index, image in enumerate(images[:20]):
        a, b, warp = pair(image, rng, True, max_blur)
        h, w = a.shape
        x, y = np.meshgrid(np.arange(12, w - 4, 24), np.arange(12, h - 4, 24))
        left = np.column_stack((x.ravel(), y.ravel()))
        right = project(left, warp)
        valid = inside(right, w, h)
        canvas = cv2.cvtColor(np.hstack((a, b)), cv2.COLOR_GRAY2BGR)
        for start, end in zip(left[valid], np.rint(right[valid]).astype(int)):
            dest = (int(end[0]) + w, int(end[1]))
            cv2.line(canvas, tuple(start), dest, (70, 185, 80), 1, cv2.LINE_AA)
            cv2.circle(canvas, dest, 2, (20, 200, 255), -1)
        if not cv2.imwrite(str(directory / f"{index:03}.png"), canvas):
            raise OSError("Could not save validation pair overlay")


def detector_targets(teacher, clean, warp):
    """Use clean pretrained detections as pseudo-labels in both warped views."""
    points = teacher.detectAndCompute(clean, top_k=300)[0]["keypoints"].cpu().numpy()
    return points, project(points, warp) if len(points) else points


def detector_loss(logits, points, rng):
    import torch
    from torch.nn.functional import cross_entropy

    _, h, w = logits.shape
    xy = np.rint(points).astype(int)
    xy = xy[inside(xy, w * 8, h * 8)]
    if not len(xy):
        return logits.sum() * 0
    cells = (xy[:, 1] // 8) * w + xy[:, 0] // 8
    positives, first = np.unique(cells, return_index=True)
    labels = (xy[first, 1] % 8) * 8 + xy[first, 0] % 8
    empty = np.setdiff1d(np.arange(h * w), positives)
    negatives = rng.choice(empty, min(len(empty), max(1, len(positives) // 8)), replace=False)
    indices = np.r_[positives, negatives]
    labels = np.r_[labels, np.full(len(negatives), 64)]
    flat = logits.permute(1, 2, 0).reshape(-1, 65)
    return cross_entropy(flat[torch.as_tensor(indices, device=logits.device)],
                         torch.as_tensor(labels, device=logits.device))


def load_images(rows, splits, width, height):
    images = {split: [] for split in splits}
    for row in rows:
        if row["split"] not in images:
            continue
        path = row["file"]
        if path.suffix == ".npy":
            image = processing_frame(np.load(path, allow_pickle=False))
        else:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unreadable image: {path}")
        images[row["split"]].append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    return images


def train_seed(args, rows, seed):
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = load_xfeat(args.xfeat_source, args.checkpoint, args.device)
    teacher = load_xfeat(args.xfeat_source, args.checkpoint, args.device) if args.train_detector else None
    output = args.output / f"seed-{seed}"
    output.mkdir(parents=True, exist_ok=False)
    paths = load_images(rows, ("train", "validation"), args.width, args.height)
    manifest = [{k: v for k, v in row.items() if k != "file"} for row in rows]
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    provenance = {"config": config, "seed": seed, "torch": torch.__version__, "opencv": cv2.__version__,
                  "base_sha256": sha256(args.checkpoint), "xfeat_revision": XFEAT_REV, "manifest": manifest,
                  "teacher": "pinned XFeat on clean frames" if teacher else "none; descriptor-only ablation"}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    # Freeze the reliability head and fine matcher. Train the detector only with teacher labels.
    for name, parameter in model.net.named_parameters():
        if name.startswith(("heatmap_head", "fine_matcher")) or (not teacher and name.startswith("keypoint_head")):
            parameter.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.net.parameters() if p.requires_grad], lr=args.lr)
    baseline = evaluate_pairs(model, paths["validation"], args.max_blur)
    best = baseline["blur"]["recall"]
    (output / "validation.jsonl").write_text(json.dumps({"step": 0, **baseline}) + "\n")
    save_validation_overlays(paths["validation"], args.max_blur, output)
    print(f"Seed {seed}, step 0: clean recall {baseline['clean']['recall']:.3f}, "
          f"blur recall {baseline['blur']['recall']:.3f}", flush=True)
    started = time.monotonic()
    interval_losses = []
    for step in range(1, args.steps + 1):
        model.net.train()
        for child in model.net.modules():
            if isinstance(child, torch.nn.BatchNorm2d):
                child.eval()
        clean_batch = [paths["train"][i] for i in rng.integers(len(paths["train"]), size=args.batch)]
        batch = [pair(image, rng, rng.random() < 0.7, args.max_blur, blur_both=rng.random() < 0.5)
                 for image in clean_batch]
        a = torch.cat([image_tensor(row[0], args.device) for row in batch])
        b = torch.cat([image_tensor(row[1], args.device) for row in batch])
        features, logits, _ = model.net(torch.cat([a, b]))
        losses = []
        for i, (_, _, warp) in enumerate(batch):
            p, q = labelled_cells(warp, args.width, args.height)
            if len(p) >= 2:
                idx = rng.choice(len(p), min(len(p), 256), replace=False)
                p_sel = torch.as_tensor(p[idx], device=args.device)
                q_sel = torch.as_tensor(q[idx], device=args.device)
                loss = descriptor_loss(features[i], features[i + len(batch)], p_sel, q_sel)
                if teacher:
                    left, right = detector_targets(teacher, clean_batch[i], warp)
                    loss += 0.25 * (detector_loss(logits[i], left, rng) +
                                    detector_loss(logits[i + len(batch)], right, rng))
                losses.append(loss)
        if not losses:
            raise RuntimeError("No usable correspondences in training batch")
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        interval_losses.append(loss.item())
        if step % args.validate_every == 0 or step == args.steps:
            scores = evaluate_pairs(model, paths["validation"], args.max_blur)
            save = output / f"step-{step}.pt"
            torch.save(model.net.state_dict(), save)
            selected = (scores["blur"]["recall"] > best
                        and scores["clean"]["recall"] >= baseline["clean"]["recall"] - 0.05)
            if selected:
                best = scores["blur"]["recall"]
                (output / "selected.json").write_text(json.dumps({"checkpoint": save.name, "sha256": sha256(save)}) + "\n")
            with (output / "validation.jsonl").open("a") as log:
                log.write(json.dumps({"step": step, "loss": float(np.mean(interval_losses)), "seconds": time.monotonic() - started,
                                      "sha256": sha256(save), "selected": selected, **scores}) + "\n")
            print(f"Seed {seed}, step {step}: loss {np.mean(interval_losses):.3f}, "
                  f"clean recall {scores['clean']['recall']:.3f}, blur recall {scores['blur']['recall']:.3f}, "
                  f"selected {selected}, {time.monotonic() - started:.1f}s", flush=True)
            interval_losses.clear()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--xfeat-source", type=Path, default=Path("runs/xfeat-upstream"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-detector", action="store_true", help="Train the detector with clean-view pseudo-labels")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-blur", type=int, default=11)
    args = parser.parse_args(argv)
    if min(args.steps, args.batch, args.validate_every) < 1 or args.max_blur < 3 or args.lr <= 0:
        parser.error("Use positive training parameters and max blur >= 3")
    if min(args.width, args.height) < 32 or args.width % 32 or args.height % 32:
        parser.error("Training dimensions must be positive multiples of 32")
    if sha256(args.checkpoint) != OFFICIAL_SHA:
        parser.error("The training seed checkpoint must match the pinned official XFeat SHA-256")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Use distinct seeds")
    rows = read_manifest(args.manifest)
    for seed in args.seeds:
        train_seed(args, rows, seed)


if __name__ == "__main__":
    main()
