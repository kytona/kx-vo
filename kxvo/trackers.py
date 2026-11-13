"""Sparse descriptor frontends for the existing landmark estimator."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

XFEAT_REV = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _digest(stream)


def _digest(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        digest.update(chunk)
    return digest.hexdigest()


def load_xfeat(source: Path, checkpoint: Path, device: str = "cpu"):
    if not checkpoint.is_file():
        raise ValueError(f"XFeat checkpoint does not exist: {checkpoint}")
    result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True)
    if result.returncode or result.stdout.strip() != XFEAT_REV or not (source / "modules/xfeat.py").is_file():
        raise ValueError(f"XFeat source must be a checkout at {XFEAT_REV}: {source}")
    try:
        import torch
    except ImportError as exc:
        raise ValueError("Install kxvo[xfeat] to use XFeat") from exc
    sys.path.insert(0, str(source.resolve()))
    try:
        module = importlib.import_module("modules.xfeat")
    finally:
        sys.path.pop(0)
    if not Path(module.__file__).is_relative_to(source.resolve()):
        raise ValueError("A different XFeat source is already loaded in this process")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = module.XFeat(weights=state).to(device)
    model.dev = torch.device(device)
    return model.eval()


class DescriptorTracker:
    def __init__(self, name, budget, checkpoint=None, source=None, device="cpu", gap_s=0.5, association="hybrid"):
        if name not in {"orb", "xfeat"}:
            raise ValueError(f"Unknown descriptor tracker: {name}")
        self.name = name
        self.budget = budget
        self.gap_s = gap_s
        self.association = association
        self.checkpoint_hash = None
        if name == "orb":
            self.model = cv2.ORB_create(nfeatures=budget * 3, edgeThreshold=16, patchSize=31)
            self.norm, self.distance = cv2.NORM_HAMMING, 65.0
        else:
            if checkpoint is None:
                raise ValueError("--checkpoint is required for XFeat")
            checkpoint = Path(checkpoint)
            self.model = load_xfeat(Path(source), checkpoint, device)
            self.checkpoint_hash = sha256(checkpoint)
            self.norm, self.distance = cv2.NORM_L2, 0.95

    def extract(self, image, mask):
        if self.name == "orb":
            keypoints, desc = self.model.detectAndCompute(image, mask)
            if desc is None:
                return np.empty((0, 2), np.float32), np.empty((0, 32), np.uint8)
            return np.array([k.pt for k in keypoints], np.float32), desc
        features = self.model.detectAndCompute(image, top_k=self.budget * 3)[0]
        points = features["keypoints"].cpu().numpy().astype(np.float32)
        px = np.rint(points).astype(int)
        px[:, 0] = np.clip(px[:, 0], 0, image.shape[1] - 1)
        px[:, 1] = np.clip(px[:, 1], 0, image.shape[0] - 1)
        keep = mask[px[:, 1], px[:, 0]] != 0
        return points[keep], features["descriptors"].cpu().numpy()[keep]

    def matches(self, prior, current):
        if not len(prior) or len(current) < 2:
            return []
        matcher = cv2.BFMatcher(self.norm)
        reverse = {m.queryIdx: m.trainIdx for m in matcher.match(current, prior)}
        return [(a.queryIdx, a.trainIdx) for a, b in matcher.knnMatch(prior, current, k=2)
                if a.distance < self.distance and a.distance < 0.85 * b.distance
                and reverse.get(a.trainIdx) == a.queryIdx]
