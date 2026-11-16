"""Prepare a fixed TartanAir V2 front-camera subset for XFeat training."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
from http.client import IncompleteRead
import io
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
from zlib import crc32
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


def route_index(archive, modality='image'):
    routes = defaultdict(list)
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.suffix.lower() != ".png" or path.parent.name != f"{modality}_{CAMERA}":
            continue
        route = path.parent.parent.name
        if re.fullmatch(r"P\d+", route):
            routes[route].append(info)
    return {route: sorted(frames, key=lambda info: info.filename) for route, frames in sorted(routes.items())}


def sample_frames(frames, count):
    if not 1 <= count <= len(frames):
        raise ValueError(f"Requested {count} distinct frames from a route of {len(frames)}")
    return [frames[((2 * index + 1) * len(frames)) // (2 * count)] for index in range(count)]


def extract_members(remote, archive, entries, workers):
    offsets = sorted(info.header_offset for info in archive.infolist())
    ends = dict(zip(offsets, offsets[1:] + [archive.start_dir]))
    missing, hashes = [], {}
    for info, path in dict(entries).items():
        if path.is_file():
            data = path.read_bytes()
            if len(data) == info.file_size and crc32(data) == info.CRC:
                hashes[info.filename] = hashlib.sha256(data).hexdigest()
                continue
        missing.append((info, path))

    def fetch_member(entry):
        info, _ = entry
        return remote.fetch(info.header_offset, ends[info.header_offset] - 1)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(missing), workers):
            batch = missing[start:start + workers]
            for (info, path), (data, offset, size) in zip(batch, pool.map(fetch_member, batch)):
                if size != remote.size:
                    raise ValueError("Archive size changed during extraction")
                remote.window, remote.window_start = data, offset
                remote.transferred += len(data)
                content = archive.read(info)
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
                    temporary.write(content)
                Path(temporary.name).replace(path)
                hashes[info.filename] = hashlib.sha256(content).hexdigest()
    return hashes


def prepare_scene(plan, root, workers, neighbor_offset=0):
    scene, difficulty = plan["scene"], plan["difficulty"]
    prefix = f"{scene}/{difficulty}"
    url = f"{SOURCE}/{prefix}/image_{CAMERA}.zip"
    rows, pairs = [], []
    remote = RemoteZip(url)
    with ZipFile(remote) as archive:
        routes = route_index(archive)
        for route, split, count in plan["routes"]:
            frames = routes[route]
            members = sample_frames(frames, count)
            directory = root / prefix / route
            entries = [(info, directory / PurePosixPath(info.filename).name) for info in members]
            if neighbor_offset:
                by_id = {int(PurePosixPath(i.filename).name.split("_")[0]): i for i in frames}
                pose = archive.getinfo(f"{prefix}/{route}/pose_{CAMERA}.txt")
                pose_path = directory / PurePosixPath(pose.filename).name
                entries.append((pose, pose_path))
                for info in members:
                    frame_id = int(PurePosixPath(info.filename).name.split("_")[0])
                    adjacent_id = frame_id + neighbor_offset
                    if adjacent_id not in by_id:
                        adjacent_id = frame_id - neighbor_offset
                    if adjacent_id not in by_id:
                        raise ValueError(f"No neighbor at offset {neighbor_offset} for {info.filename}")
                    neighbor = by_id[adjacent_id]
                    neighbor_path = directory / PurePosixPath(neighbor.filename).name
                    entries.append((neighbor, neighbor_path))
                    pairs.append({"group": f"{prefix}/{route}", "split": split,
                                  "frame_a": frame_id, "frame_b": adjacent_id,
                                  "rgb_a": (directory / PurePosixPath(info.filename).name).relative_to(root).as_posix(),
                                  "rgb_b": neighbor_path.relative_to(root).as_posix(),
                                  "pose": pose_path.relative_to(root).as_posix()})
            hashes = extract_members(remote, archive, entries, workers)
            for info in members:
                path = directory / PurePosixPath(info.filename).name
                rows.append({"path": path.relative_to(root).as_posix(), "split": split,
                             "group": f"{prefix}/{route}", "rights": RIGHTS, "camera": CAMERA,
                             "source": url, "member": info.filename, "sha256": hashes[info.filename]})
            if neighbor_offset:
                for pair in pairs[-count:]:
                    pair["pose_sha256"] = hashes[pose.filename]
                    for view in ("a", "b"):
                        member = by_id[pair[f"frame_{view}"]]
                        pair[f"rgb_{view}_sha256"] = hashes[member.filename]
            print(f"{prefix}/{route}: {count} {split} frames", flush=True)
    transferred = remote.transferred
    if pairs:
        depth_url = f"{SOURCE}/{prefix}/depth_{CAMERA}.zip"
        depth = RemoteZip(depth_url)
        with ZipFile(depth) as archive:
            entries = []
            for pair in pairs:
                pair["depth_source"] = depth_url
                for view in ("a", "b"):
                    name = f"{pair[f'frame_{view}']:06}_{CAMERA}_depth.png"
                    info = archive.getinfo(f"{pair['group']}/depth_{CAMERA}/{name}")
                    path = root / pair["group"] / name
                    pair[f"depth_{view}"] = path.relative_to(root).as_posix()
                    entries.append((info, path))
            hashes = extract_members(depth, archive, entries, workers)
            for pair in pairs:
                for view in ("a", "b"):
                    source = PurePosixPath(pair[f"depth_{view}"])
                    pair[f"depth_{view}_sha256"] = hashes[str(source.parent / f"depth_{CAMERA}" / source.name)]
        transferred += depth.transferred
    return rows, pairs, transferred


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def prepare(plan, output, workers, neighbor_offset):
    output.parent.mkdir(parents=True, exist_ok=True)
    root = output.with_name(f".{output.name}.partial")
    lock = output.with_name(f".{output.name}.lock")
    config = {"revision": REVISION, "plan": plan, "neighbor_offset": neighbor_offset}
    with lock.open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            raise FileExistsError(output)
        root.mkdir(exist_ok=True)
        state = root / "download-plan.json"
        if state.exists() and json.loads(state.read_text()) != config:
            raise ValueError(f"Existing partial dataset in {root} uses a different plan")
        state.write_text(json.dumps(config, indent=2) + "\n")
        with ThreadPoolExecutor(max_workers=min(4, len(plan))) as scenes:
            extracted = list(scenes.map(lambda scene: prepare_scene(scene, root, workers, neighbor_offset), plan))
        rows = [row for scene_rows, _, _ in extracted for row in scene_rows]
        write_csv(root / "manifest.csv", rows)
        from .train import read_manifest
        read_manifest(root / "manifest.csv", required_splits=())
        pairs = [pair for _, scene_pairs, _ in extracted for pair in scene_pairs]
        if pairs:
            write_csv(root / "pairs.csv", pairs)
            (root / "calibration.json").write_text(json.dumps({
                "camera": CAMERA, "width": 640, "height": 640, "fx": 320, "fy": 320,
                "cx": 320, "cy": 320, "distortion": [0, 0, 0, 0], "fps": 10,
                "pose_convention": "TartanAir camera pose in NED, x y z qx qy qz qw",
                "source": "https://tartanair.org/modalities.html"
            }, indent=2) + "\n")
        provenance = {**config, "source": SOURCE, "rights": RIGHTS,
                      "bytes_transferred_this_attempt": sum(size for _, _, size in extracted)}
        (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        root.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--neighbor-offset", type=int, default=0,
                        help="Also fetch nearby RGB, both depth maps and trajectory pose at this frame offset")
    args = parser.parse_args()
    if args.workers < 1 or args.neighbor_offset < 0 or args.output.exists():
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
    prepare(plan, args.output, args.workers, args.neighbor_offset)

if __name__ == "__main__":
    main()
