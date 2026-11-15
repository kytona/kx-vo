"""Prepare a fixed TartanAir V2 front-camera subset for XFeat training."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from http.client import IncompleteRead
import io
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile

REVISION = "0d2d145e973832742a2aaa04b7d2ebffc8d82817"
SOURCE = f"https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/{REVISION}"
RIGHTS = "CC BY 4.0 (https://tartanair.org/)"
CAMERA = "lcam_front"


class RemoteZip(io.RawIOBase):
    """Seek in an immutable HTTP archive without downloading the full scene."""

    def __init__(self, url):
        self.url = url
        self.position = 0
        self.window, self.window_start, self.size = self.fetch(None, 65536)
        self.transferred = len(self.window)

    def fetch(self, start, end):
        span = f"-{end}" if start is None else f"{start}-{end}"
        request = Request(f"{self.url}?download=true&range={span}", headers={"Range": f"bytes={span}"})
        for attempt in range(4):
            try:
                with urlopen(request, timeout=30) as response:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if response.status != 206 or not match:
                        raise ValueError(f"Server did not honour byte range {span}")
                    low, high, size = map(int, match.groups())
                    expected = (max(0, size - end), size - 1) if start is None else (start, end)
                    if (low, high) != expected:
                        raise ValueError(f"Unexpected response range for {span}: {match[0]}")
                    expected_length = high - low + 1
                    data = response.read(expected_length + 1)
                    if len(data) != expected_length:
                        raise OSError(f"Incomplete byte range {span}")
                    return data, low, size
            except (OSError, IncompleteRead):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        position = base + offset
        if position < 0:
            raise OSError("Seek before archive start")
        self.position = position
        return position

    def read(self, count=-1):
        start = self.position
        stop = self.size if count < 0 else min(start + count, self.size)
        if stop <= start:
            return b""
        offset = start - self.window_start
        if 0 <= offset and offset + stop - start <= len(self.window):
            data = self.window[offset:offset + stop - start]
        else:
            data, _, _ = self.fetch(start, stop - 1)
            self.transferred += len(data)
        self.position = stop
        return data


def route_index(archive):
    routes = defaultdict(list)
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.suffix.lower() != ".png" or path.parent.name != f"image_{CAMERA}":
            continue
        route = path.parent.parent.name
        if re.fullmatch(r"P\d+", route):
            routes[route].append(info)
    return {route: sorted(frames, key=lambda info: info.filename) for route, frames in sorted(routes.items())}


def sample_frames(frames, count):
    if not 1 <= count <= len(frames):
        raise ValueError(f"Requested {count} distinct frames from a route of {len(frames)}")
    return [frames[((2 * index + 1) * len(frames)) // (2 * count)] for index in range(count)]


def prepare_scene(plan, root, workers):
    scene = plan["scene"]
    difficulty = plan["difficulty"]
    url = f"{SOURCE}/{scene}/{difficulty}/image_{CAMERA}.zip"
    remote = RemoteZip(url)
    rows = []
    with ZipFile(remote) as archive:
        routes = route_index(archive)
        offsets = sorted(info.header_offset for info in archive.infolist())
        ends = dict(zip(offsets, offsets[1:] + [archive.start_dir]))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for route, split, count in plan["routes"]:
                members = sample_frames(routes[route], count)

                def fetch_member(info):
                    start = info.header_offset
                    # Include the local ZIP header and compressed payload.
                    return remote.fetch(start, ends[start] - 1)

                directory = root / scene / difficulty / route
                directory.mkdir(parents=True)
                for start in range(0, len(members), workers):
                    batch = members[start:start + workers]
                    for info, (data, offset, size) in zip(batch, pool.map(fetch_member, batch)):
                        if size != remote.size:
                            raise ValueError("Archive size changed during extraction")
                        remote.window, remote.window_start = data, offset
                        remote.transferred += len(data)
                        image = archive.read(info)  # validates ZIP member CRC
                        path = directory / PurePosixPath(info.filename).name
                        path.write_bytes(image)
                        rows.append({"path": path.relative_to(root).as_posix(), "split": split,
                                     "group": f"{scene}/{difficulty}/{route}", "rights": RIGHTS,
                                     "camera": CAMERA, "source": url, "member": info.filename,
                                     "sha256": hashlib.sha256(image).hexdigest()})
                print(f"{scene}/{difficulty}/{route}: {count} {split} frames", flush=True)
    return rows, remote.transferred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.output.exists():
        parser.error("Use positive workers and a new output directory")
    plan = json.loads(args.plan.read_text())
    if not plan:
        parser.error("The download plan must contain at least one scene")
    seen = set()
    for scene in plan:
        name, difficulty = scene["scene"], scene["difficulty"]
        if not re.fullmatch(r"[A-Za-z0-9_]+", name) or difficulty not in {"Data_easy", "Data_hard"}:
            parser.error(f"Invalid scene or difficulty: {name}/{difficulty}")
        for route, split, count in scene["routes"]:
            group = (name, difficulty, route)
            if (not re.fullmatch(r"P[0-9]+", route) or split not in {"train", "validation", "test"}
                    or not isinstance(count, int) or count < 1 or group in seen):
                parser.error(f"Invalid or duplicate route: {group}")
            seen.add(group)
    if not seen:
        parser.error("The download plan must contain at least one route")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tartanair-", dir=args.output.parent) as temporary:
        root = Path(temporary) / "subset"
        root.mkdir()
        with ThreadPoolExecutor(max_workers=min(4, len(plan))) as scenes:
            extracted = list(scenes.map(lambda scene: prepare_scene(scene, root, args.workers), plan))
        rows = [row for scene_rows, _ in extracted for row in scene_rows]
        with (root / "manifest.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        from .train import read_manifest
        read_manifest(root / "manifest.csv")
        provenance = {"revision": REVISION, "source": SOURCE, "rights": RIGHTS,
                      "plan": plan, "bytes_transferred": sum(size for _, size in extracted)}
        (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        root.rename(args.output)


if __name__ == "__main__":
    main()
