"""Video input: HDMI capture dongle, a video file, or a recorded run."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


class VideoSource:
    """Grayscale frames resized to the processing width, with capture times.

    ``latency_s`` is subtracted from the host receive time. For the HDZero
    path (camera, VTX, BoxPro, HDMI dongle) measure it once by filming a
    millisecond clock on screen next to its live feed, rather than guessing.
    """

    def __init__(self, src: str, width: int, latency_s: float = 0.0):
        index = int(src) if src.isdigit() else None
        if index is not None:
            api = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(index, api)
        else:
            self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video source {src!r}")
        self.is_file = index is None
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = width
        self.latency_s = latency_s
        self.n = 0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None, None
        t = (self.n / self.fps) if self.is_file else time.monotonic() - self.latency_s
        self.n += 1
        return t, processing_frame(frame, self.width), frame

    def close(self):
        self.cap.release()


def processing_frame(frame, width=None):
    h, w = frame.shape[:2]
    if width is not None and width != w:
        frame = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


class Recorder:
    """Saves source frames and a JSONL log so a run can be replayed offline."""

    def __init__(self, out_dir: str | Path, save_frames: bool = True, frame_format: str = "png"):
        if frame_format not in {"png", "npy"}:
            raise ValueError("Unsupported frame format")
        self.frame_format = frame_format
        self.dir = Path(out_dir)
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self.log = (self.dir / "log.jsonl").open("x")
        self.save_frames = save_frames
        self.i = 0

    def write(self, t: float, frame, record: dict):
        name = f"{self.i:06d}.{self.frame_format}"
        if self.save_frames and frame is not None:
            path = self.dir / "frames" / name
            if self.frame_format == "npy":
                np.save(path, frame, allow_pickle=False)
            elif not cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError(f"Could not save frame {name}")
        record = {"i": self.i, "t": t, "frame": name if self.save_frames else None, **record}
        self.log.write(json.dumps(record) + "\n")
        self.i += 1

    def close(self):
        self.log.close()


def replay_records(run_dir: str | Path, width=None):
    """Yield (t, gray, range_m, att) from a recorded run directory."""
    run_dir = Path(run_dir)
    with (run_dir / "log.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("frame"):
                continue
            path = run_dir / "frames" / rec["frame"]
            frame = np.load(path, allow_pickle=False) if path.suffix == ".npy" else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise ValueError(f"Missing or unreadable recorded frame: {rec['frame']}")
            att = tuple(rec["att"]) if rec.get("att") else None
            yield rec["t"], processing_frame(frame, width), rec.get("range_m"), att
