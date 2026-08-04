import argparse
import time

import numpy as np

from lane_simulator import LaneSimulator


parser = argparse.ArgumentParser()
parser.add_argument("--environments", type=int, default=4096)
parser.add_argument("--steps", type=int, default=1000)
args = parser.parse_args()

environment = LaneSimulator(args.environments)
actions = np.zeros(args.environments, dtype=np.int64)
started = time.perf_counter()
for _ in range(args.steps):
    environment.step(actions)
elapsed = time.perf_counter() - started
transitions = args.environments * args.steps
print({"transitions": transitions, "seconds": elapsed, "transitions_per_second": transitions / elapsed})
