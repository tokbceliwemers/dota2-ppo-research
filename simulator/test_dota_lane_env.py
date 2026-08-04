import json
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from dota_lane_env import ACTION_DIM, ATTACK_ACTION, OBSERVATION_DIM, DotaTerrainLaneEnv, TerrainLaneConfig
from terrain import TerrainHeightmap


def write_terrain(directory: Path, heights: np.ndarray, *, map_min: float = -10.0, map_max: float = 10.0) -> None:
    np.save(directory / "dota_heightmap.npy", heights.astype(np.float32))
    (directory / "dota_heightmap.json").write_text(json.dumps({
        "map_min": map_min, "map_max": map_max, "raw_height_min": 0.0, "raw_height_max": 100.0,
    }), encoding="utf-8")


class TerrainHeightmapTest(unittest.TestCase):
    def test_bilinear_height_and_slope(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            write_terrain(directory, np.array(((0, 0, 0), (0, .5, 1), (0, 1, 1)), dtype=np.float32))
            terrain = TerrainHeightmap.from_data_directory(directory)
            self.assertAlmostEqual(float(terrain.height(np.array((0.0, 0.0)))), 50.0, places=4)
            self.assertGreater(float(terrain.slope(np.array((0.0, 0.0)))), 0.0)
            self.assertFalse(terrain.traversable(np.array((0.0, 0.0)), np.array((20.0, 0.0)), 10.0))


class DotaTerrainLaneEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        # Flat terrain isolates the lane contract from terrain-specific tests.
        write_terrain(self.directory, np.zeros((32, 32), dtype=np.float32), map_min=-2_000.0, map_max=2_000.0)
        self.config = TerrainLaneConfig(spawn_margin=1_000.0, horizon_steps=3, max_ground_slope=0.1)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_gymnasium_contract_and_seeded_reset(self) -> None:
        first = DotaTerrainLaneEnv(self.directory, self.config)
        second = DotaTerrainLaneEnv(self.directory, self.config)
        observation, info = first.reset(seed=17)
        other, _ = second.reset(seed=17)
        self.assertEqual(observation.shape, (OBSERVATION_DIM,))
        self.assertTrue(first.observation_space.contains(observation))
        self.assertTrue(np.array_equal(observation, other))
        self.assertEqual(info["action_mask"].shape, (ACTION_DIM,))
        self.assertIsInstance(first, gym.Env)
        check_env(DotaTerrainLaneEnv(self.directory, self.config), skip_render_check=True)

    def test_attack_is_range_gated_and_health_bar_is_visible_only(self) -> None:
        env = DotaTerrainLaneEnv(self.directory, self.config)
        env.reset(seed=4)
        env.hero_xy = np.array((-900.0, -900.0), dtype=np.float32)
        env.enemy_xy[:] = np.array((900.0, 900.0), dtype=np.float32)
        before = env.enemy_hp.copy()
        observation, reward, _, _, info = env.step(ATTACK_ACTION)
        self.assertLessEqual(float(env.enemy_hp.sum()), float(before.sum()))  # allied pressure only
        self.assertFalse(bool(info["action_mask"][ATTACK_ACTION]))
        self.assertLess(reward, 0.0)
        self.assertIn(observation[13], np.arange(21, dtype=np.float32) / 20.0)

    def test_episode_terminates_when_wave_is_cleared(self) -> None:
        env = DotaTerrainLaneEnv(self.directory, self.config)
        env.reset(seed=3)
        env.enemy_hp[:] = 0.0
        _, _, terminated, truncated, _ = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()
