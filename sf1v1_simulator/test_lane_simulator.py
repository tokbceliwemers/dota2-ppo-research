import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from lane_simulator import ACTION_DIM, ATTACK_ACTION, OBSERVATION_DIM, LaneConfig, LaneSimulator


class LaneSimulatorTest(unittest.TestCase):
    def test_contract_and_determinism(self) -> None:
        first = LaneSimulator(32, seed=9)
        second = LaneSimulator(32, seed=9)
        self.assertEqual(first.observation().shape, (32, OBSERVATION_DIM))
        self.assertTrue(np.array_equal(first.observation(), second.observation()))
        self.assertEqual(first.action_mask().shape, (32, ACTION_DIM))

    def test_attack_is_range_gated(self) -> None:
        simulator = LaneSimulator(1, LaneConfig(hero_attack_range=1.0), seed=3)
        simulator.creep_xy[:] = 900.0
        before = simulator.creep_hp.copy()
        simulator.step(np.array([ATTACK_ACTION]))
        self.assertLess(simulator.creep_hp[0], before[0])  # passive lane damage only

    def test_terminal_resets_state(self) -> None:
        simulator = LaneSimulator(4, LaneConfig(horizon_steps=2), seed=5)
        _, _, done, _ = simulator.step(np.zeros(4, dtype=np.int64))
        self.assertFalse(done.any())
        _, _, done, _ = simulator.step(np.zeros(4, dtype=np.int64))
        self.assertTrue(done.all())
        self.assertTrue((simulator.steps == 0).all())

    def test_uses_only_measured_calibration_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps({"source": "local_lane_calibration", "measured_lane_config": {
                "hero_attack_range": 475.0, "hero_move_speed": 310.0, "ignored": 1}}), encoding="utf-8")
            config = LaneConfig.from_calibration(path)
        self.assertEqual(config.hero_attack_range, 475.0)
        self.assertEqual(config.hero_move_speed, 310.0)
        self.assertEqual(config.hero_attack_cooldown, LaneConfig().hero_attack_cooldown)


if __name__ == "__main__":
    unittest.main()
