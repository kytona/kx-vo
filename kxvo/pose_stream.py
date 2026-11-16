"""Timestamped local NED poses for the simulator's ExternalNav input."""

from bisect import bisect_right
import json
import math
from types import SimpleNamespace

import numpy as np


class PoseStream:
    def __init__(self, path):
        self.times = []
        self.poses = []
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                row = json.loads(line)
                p, rpy = np.asarray(row["p"], dtype=float), np.asarray(row["rpy"], dtype=float)
                t = float(row["t"])
                sigma = float(row.get("sigma_m", 0.05))
                reset = row.get("reset", 0)
                valid = row.get("valid", True)
                if (not math.isfinite(t) or t < 0 or (self.times and t <= self.times[-1])
                        or p.shape != (3,) or rpy.shape != (3,) or not np.isfinite([p, rpy]).all()
                        or not math.isfinite(sigma) or sigma <= 0
                        or type(reset) is not int or not 0 <= reset <= 255 or type(valid) is not bool):
                    raise ValueError(f"Invalid pose stream row {number} in {path}")
                self.times.append(t)
                self.poses.append(SimpleNamespace(p_WB=p, rpy=rpy.tolist(), valid=valid,
                                                  reset_counter=reset, pos_sigma_m=sigma,
                                                  status="recorded pose"))
        if not self.times:
            raise ValueError("Pose stream is empty")

    def sample(self, elapsed, max_age=0.2):
        index = bisect_right(self.times, elapsed) - 1
        if index < 0 or elapsed - self.times[index] > max_age:
            return None, None
        return self.times[index], self.poses[index]
