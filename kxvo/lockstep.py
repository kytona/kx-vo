"""Verify ArduCopter's JSON clock against a stationary external physics fixture."""

import argparse
import hashlib
import json
from pathlib import Path
import signal
import socket
import struct
import subprocess

from .sitl import free_port
from .trackers import sha256


def servo_frame(payload):
    if len(payload) not in (40, 72):
        raise ValueError(f"Unexpected servo packet size {len(payload)}")
    magic, rate, index = struct.unpack_from("<HHI", payload)
    if (magic, len(payload)) not in ((18458, 40), (29569, 72)) or not rate:
        raise ValueError("Invalid ArduPilot JSON servo packet")
    return rate, index


def probe(arducopter, output, ticks, pause_s, instance):
    output.mkdir(parents=True)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as backend, (output / "autopilot.log").open("w") as log:
        backend.bind(("127.0.0.1", 0))
        backend.settimeout(10)
        port = backend.getsockname()[1]
        process = subprocess.Popen([
            str(arducopter.resolve()), "--model", "JSON:127.0.0.1", "--sim-port-out", str(port),
            "--speedup", "1", "--serial0", f"udpclient:127.0.0.1:{free_port()}", "-I", str(instance), "-w",
        ], cwd=output, stdout=log, stderr=subprocess.STDOUT)
        try:
            packet, sender = backend.recvfrom(128)
            rate, first = servo_frame(packet)
            before_pause = ticks // 2
            pauses = []
            servos = hashlib.sha256()
            for step in range(ticks):
                if step == before_pause:
                    backend.settimeout(pause_s)
                    paused = True
                    try:
                        pending, peer = backend.recvfrom(128)
                        paused = peer == sender and servo_frame(pending) == (rate, first + step)
                    except TimeoutError:
                        pass
                    finally:
                        backend.settimeout(10)
                    pauses.append(paused)
                seconds = (step + 1) / rate
                sensor = {"timestamp": seconds, "imu": {"gyro": [0, 0, 0], "accel_body": [0, 0, -9.80665]},
                          "position": [0, 0, 0], "attitude": [0, 0, 0], "velocity": [0, 0, 0],
                          "no_lockstep": False}
                message = ("\n" + json.dumps(sensor, separators=(",", ":")) + "\n").encode()
                backend.sendto(message, sender)
                packet, peer = backend.recvfrom(128)
                if peer != sender or servo_frame(packet) != (rate, first + step + 1):
                    raise RuntimeError(f"ArduCopter did not advance exactly one servo counter at tick {step}")
                servos.update(packet)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    result = {"passed": all(pauses), "ticks": ticks, "rate_hz": rate, "pause_wall_s": pause_s,
              "held_at_tick": before_pause, "first_counter": first, "final_counter": first + ticks,
              "servo_packets_sha256": servos.hexdigest(), "arducopter_sha256": sha256(arducopter)}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arducopter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=2000)
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument("--instance", type=int, default=7)
    args = parser.parse_args()
    if args.ticks < 2 or not 0 < args.pause < 1:
        parser.error("Use at least two ticks and a pause shorter than the servo retransmit interval")
    result = probe(args.arducopter, args.output, args.ticks, args.pause, args.instance)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
