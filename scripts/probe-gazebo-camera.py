"""Read and verify three Gazebo Image messages using Gazebo Harmonic's Python bindings."""

import argparse
import hashlib
import json
from pathlib import Path
from queue import Queue
import struct
import time

from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--relay", action="store_true", help="Publish the latest RGB frame atomically")
    parser.add_argument("--topic", default="/kxvo/front_camera")
    args = parser.parse_args()
    node = Node()
    frames = Queue()
    if args.relay:
        temp = args.output.with_suffix(".tmp")

        def relay(frame):
            if (frame.width, frame.height, frame.step, len(frame.data)) != (320, 240, 960, 230400):
                return
            stamp = frame.header.stamp.sec * 10**9 + frame.header.stamp.nsec
            payload = struct.pack("<QQ", stamp, time.monotonic_ns()) + frame.data
            temp.write_bytes(payload)
            temp.replace(args.output)

        if not node.subscribe(Image, args.topic, relay):
            raise RuntimeError(f"Could not subscribe to {args.topic}")
        while True:
            time.sleep(1)
    if not node.subscribe(Image, args.topic, frames.put):
        raise RuntimeError(f"Could not subscribe to {args.topic}")
    captured = []
    for _ in range(3):
        frame = frames.get(timeout=20)
        if frame.width != 320 or frame.height != 240 or frame.step != 960 or len(frame.data) != 230400:
            raise ValueError("Unexpected Gazebo RGB frame geometry")
        timestamp_ns = frame.header.stamp.sec * 10**9 + frame.header.stamp.nsec
        captured.append({"timestamp_ns": timestamp_ns, "sha256": hashlib.sha256(frame.data).hexdigest()})
    if not all(a["timestamp_ns"] < b["timestamp_ns"] for a, b in zip(captured, captured[1:])):
        raise RuntimeError("Camera timestamps did not advance")
    result = {"topic": args.topic, "width": 320, "height": 240, "frames": captured}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
