"""Video input: HDMI capture dongle, a video file, or a recorded run."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2


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
        h, w = frame.shape[:2]
        if w != self.width:
            frame = cv2.resize(frame, (self.width, int(round(h * self.width / w))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return t, gray, frame

    def close(self):
        self.cap.release()


class Recorder:
    """Saves processed frames and a JSONL log so a run can be replayed offline."""

    def __init__(self, out_dir: str | Path, save_frames: bool = True):
        self.dir = Path(out_dir)
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self.log = (self.dir / "log.jsonl").open("a")
        self.save_frames = save_frames
        self.i = 0

    def write(self, t: float, gray, record: dict):
        name = f"{self.i:06d}.png"
        if self.save_frames and gray is not None:
            cv2.imwrite(str(self.dir / "frames" / name), gray)
        record = {"i": self.i, "t": t, "frame": name if self.save_frames else None, **record}
        self.log.write(json.dumps(record) + "\n")
        self.i += 1

    def close(self):
        self.log.close()


def replay_records(run_dir: str | Path):
    """Yield (t, gray, range_m, att) from a recorded run directory."""
    run_dir = Path(run_dir)
    with (run_dir / "log.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("frame"):
                continue
            gray = cv2.imread(str(run_dir / "frames" / rec["frame"]), cv2.IMREAD_GRAYSCALE)
            att = tuple(rec["att"]) if rec.get("att") else None
            yield rec["t"], gray, rec.get("range_m"), att
