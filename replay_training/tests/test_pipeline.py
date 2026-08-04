from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dota_ppo.actions import ACTION_DIM, ACTION_IDS
from dota_ppo.bridge import PolicyBridge
from dota_ppo.calibration import SCHEMA_VERSION, analyze_events
from dota_ppo.comparison import compare_rollout_sets
from dota_ppo.controls import canonicalize_input
from dota_ppo.curriculum import synthetic_lane_expert
from dota_ppo.data import (CURRENT_LOCAL_OBSERVATION_VERSION, CURRENT_LOCAL_REWARD_VERSION,
                           Rollouts, Trajectories, checkpoint_sha256, load_rollout_metadata,
                           load_rollouts, load_trajectories, merge_rollouts, save_rollouts)
from dota_ppo.evaluation import evaluate_rollouts
from dota_ppo.input_logs import canonicalize_jsonl, join_orders_to_states
from dota_ppo.replays import build_replay_dataset
from dota_ppo.train import PPOConfig, _gae, behavior_clone, ppo_update, ppo_update_headless_lane
from dota_ppo.headless_lane import (HeadlessLaneConfig, SOURCE as HEADLESS_SOURCE,
                                    HeadlessLane, TorchHeadlessLane, collect_headless_rollout,
                                    train_headless_lane)
from dota_ppo.model import ActorCritic
from dota_ppo.observations import OBSERVATION_DIM, OBSERVATION_VERSION, health_bar_fraction
from dota_ppo.supervisor import SupervisorConfig, run_supervisor, validate_rollout


def _replay_data() -> Trajectories:
    return Trajectories(np.zeros((8, OBSERVATION_DIM), np.float32), np.arange(8) % ACTION_DIM,
                        np.ones(8, np.float32), np.array([False] * 7 + [True]), "gem_replay_approximate")


def test_behavior_cloning_writes_a_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    metrics = behavior_clone(_replay_data(), ACTION_DIM, torch.device("cpu"), checkpoint, epochs=1, batch_size=4)
    assert checkpoint.exists()
    assert metrics["steps"] == 8


def test_synthetic_lane_expert_contains_attack_labels() -> None:
    data = synthetic_lane_expert(256, seed=4)
    assert data.observations.shape == (256, OBSERVATION_DIM)
    assert (data.actions == ACTION_IDS["attack"]).any()


def test_ppo_rejects_approximate_replay_actions() -> None:
    data = _replay_data()
    rollouts = Rollouts(data.observations, data.actions, data.rewards, data.dones, data.source,
                        np.zeros(8, np.float32), np.zeros(8, np.float32))
    try:
        ppo_update(rollouts, ActorCritic(OBSERVATION_DIM, ACTION_DIM), torch.device("cpu"), PPOConfig(epochs=1, minibatch_size=4))
    except ValueError as error:
        assert "exact on-policy" in str(error)
    else:
        raise AssertionError("PPO accepted replay-inferred labels")


def test_ppo_updates_exact_local_rollout() -> None:
    data = _replay_data()
    rollouts = Rollouts(data.observations, data.actions, data.rewards, data.dones, "local_instrumented_lobby",
                        np.zeros(8, np.float32), np.zeros(8, np.float32), np.ones((8, ACTION_DIM), dtype=bool))
    metrics = ppo_update(rollouts, ActorCritic(OBSERVATION_DIM, ACTION_DIM), torch.device("cpu"), PPOConfig(epochs=1, minibatch_size=4))
    assert set(metrics) == {"policy_loss", "value_loss", "entropy", "approx_kl"}


def test_v4_lane_addon_enforces_fixed_shadow_fiend_progression() -> None:
    addon = Path(__file__).parents[1] / "dota_addon" / "scripts" / "vscripts"
    scenario = (addon / "rl_lane_scenario.lua").read_text(encoding="utf-8")
    entry = (addon / "addon_game_mode.lua").read_text(encoding="utf-8")
    bridge = (addon / "rl_ppo_bridge.lua").read_text(encoding="utf-8")
    assert 'SetModifyExperienceFilter(Dynamic_Wrap(GameMode, "FilterModifyExperience"), self)' in entry
    assert 'filter["hero_entindex_const"] == hero:entindex()' in entry
    assert 'modifier_nevermore_necromastery' in scenario
    assert 'ability:SetLevel(0)' in scenario
    assert 'modifier:SetStackCount(0)' in scenario
    assert 'lane_wave_clear_v4_fixed_progression' in bridge
    assert 'local OBSERVATION_DIM = 25' in bridge
    assert 'GetAttackTarget' in bridge
    assert 'GetLastAttackTime' in bridge


def test_health_bar_observation_is_quantized() -> None:
    bars = health_bar_fraction(np.array([0.0, 0.024, 0.026, 0.99, 1.0], dtype=np.float32))
    np.testing.assert_allclose(bars, np.array([0.0, 0.0, 0.05, 1.0, 1.0], dtype=np.float32))


def test_headless_lane_uses_live_compatible_shapes_and_masks() -> None:
    environment = HeadlessLane(16, seed=3, horizon=8)
    observations, masks = environment.observation(), environment.action_masks()
    assert observations.shape == (16, OBSERVATION_DIM)
    assert masks.shape == (16, ACTION_DIM)
    assert masks[:, 0].all()
    assert masks[:, ACTION_IDS["ability_1"]].sum() == 0


def test_headless_lane_ends_when_allied_pressure_kills_the_creep() -> None:
    environment = HeadlessLane(4, seed=3, horizon=8)
    environment.creep_hp[:] = 0.001
    environment.allied_pressure[:] = 1.0
    _obs, rewards, done, _masks = environment.step(np.zeros(4, dtype=np.int64))
    assert done.all()
    assert not (rewards >= 0.75).any()


def test_torch_headless_lane_has_live_compatible_shapes() -> None:
    environment = TorchHeadlessLane(4, torch.device("cpu"), seed=3, horizon=8)
    observations, masks = environment.observation(), environment.action_masks()
    assert observations.shape == (4, OBSERVATION_DIM)
    assert masks.shape == (4, ACTION_DIM)
    assert masks[:, 0].all()
    _obs, _rewards, _done, after_masks = environment.step(torch.zeros(4, dtype=torch.long))
    assert after_masks[:, ACTION_IDS["ability_1"]].sum() == 0


def test_headless_lane_loads_only_measured_calibration_fields(tmp_path: Path) -> None:
    report = tmp_path / "calibration.json"
    report.write_text('{"source":"local_lane_calibration","measured_lane_config":{"tick_seconds":0.3,"hero_move_speed":305,"hero_attack_range":525,"unknown":2}}', encoding="utf-8")
    config = HeadlessLaneConfig.from_calibration(report)
    assert config.tick_seconds == 0.3
    assert config.hero_move_speed == 305
    assert config.hero_attack_range == 525
    assert config.hero_attack_cooldown == HeadlessLaneConfig().hero_attack_cooldown


def test_headless_lane_pretraining_is_separate_from_real_ppo(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bootstrap.pt"
    behavior_clone(_replay_data(), ACTION_DIM, torch.device("cpu"), checkpoint, epochs=1, batch_size=4)
    model = ActorCritic(OBSERVATION_DIM, ACTION_DIM)
    rollout = collect_headless_rollout(model, 16, 8, seed=5)
    assert rollout.source == HEADLESS_SOURCE
    assert len(rollout.actions) == 128
    assert rollout.dones.reshape(16, 8)[:, -1].all()
    assert rollout.action_masks[np.arange(len(rollout.actions)), rollout.actions].all()
    ppo_update_headless_lane(rollout, model, torch.device("cpu"), PPOConfig(epochs=1, minibatch_size=32))
    output = tmp_path / "headless.pt"
    result = train_headless_lane(checkpoint, output, torch.device("cpu"), updates=1, environments=16, horizon=8, epochs=1)
    assert output.exists()
    assert result["source"] == HEADLESS_SOURCE
    assert result["samples"] == 128


def test_gae_respects_environment_major_trajectory_boundaries() -> None:
    rewards = torch.tensor([1.0, 2.0, 10.0, 20.0])
    dones = torch.tensor([False, True, False, True])
    advantages, returns = _gae(rewards, dones, torch.zeros(4), gamma=1.0, gae_lambda=1.0)
    torch.testing.assert_close(advantages, torch.tensor([3.0, 2.0, 30.0, 20.0]))
    torch.testing.assert_close(returns, advantages)


def test_parquet_dataset_has_compatible_shapes(tmp_path: Path) -> None:
    match = tmp_path / "123"; match.mkdir()
    pd.DataFrame({"player_id": [1, 1, 1], "hero_name": ["npc_dota_hero_nevermore"] * 3,
                  "team": [0, 0, 0], "tick": [0, 30, 60], "x": [0, 200, 400], "y": [0, 0, 0]}).to_parquet(match / "positions.parquet")
    pd.DataFrame({"player_id": [1, 1], "tick": [0, 60], "net_worth": [600, 700], "lh": [0, 1], "xp": [0, 50]}).to_parquet(match / "players_minute.parquet")
    output = tmp_path / "dataset.npz"
    result = build_replay_dataset([match], output, "nevermore")
    loaded = load_trajectories(output)
    assert result["steps"] == 2
    assert loaded.observations.shape == (2, OBSERVATION_DIM)
    assert loaded.actions.shape == (2,)


def test_personal_hotkey_maps_to_canonical_inventory_key() -> None:
    label = canonicalize_input({"tick": 12, "player_id": 1, "raw_key": "I", "order": "use_item", "inventory_slot": 1})
    assert label.raw_key == "I"
    assert label.action_label == "item_1"
    assert label.canonical_key == "Z"
    assert label.action_id is not None


def test_input_log_export_preserves_audit_key(tmp_path: Path) -> None:
    source = tmp_path / "inputs.jsonl"
    source.write_text('{"tick":12,"player_id":1,"raw_key":"I","order":"use_item","inventory_slot":1}\n', encoding="utf-8")
    output = tmp_path / "labels.parquet"
    result = canonicalize_jsonl(source, output)
    row = pd.read_parquet(output).iloc[0]
    assert result["records"] == 1
    assert row["raw_key"] == "I"
    assert row["canonical_key"] == "Z"
    assert row["action_id"] >= 0


def test_bridge_writes_exact_ppo_rollout(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    behavior_clone(_replay_data(), ACTION_DIM, torch.device("cpu"), checkpoint, epochs=1, batch_size=4)
    output = tmp_path / "rollout.npz"
    bridge = PolicyBridge(checkpoint, output, "cpu")
    decision = bridge.act({"observation": [0.0] * OBSERVATION_DIM, "action_mask": [True] + [False] * (ACTION_DIM - 1), "game_time": 12.25,
                           "reward_version": "test_v2"})
    bridge.transition({"decision_id": decision["decision_id"], "reward": 1.0, "done": True})
    result = bridge.flush()
    rollout = load_rollouts(output)
    assert result["steps"] == 1
    assert rollout.source == "local_instrumented_lobby"
    assert rollout.old_log_probs.shape == (1,)
    assert rollout.action_masks.shape == (1, ACTION_DIM)
    assert rollout.game_times.tolist() == [12.25]
    assert load_rollout_metadata(output)["reward_version"] == "test_v2"
    assert load_rollout_metadata(output)["observation_version"] == OBSERVATION_VERSION
    assert load_rollout_metadata(output)["policy_checkpoint_sha256"] == checkpoint_sha256(checkpoint)
    assert bridge.health()["action_requests"] == 1
    assert bridge.health()["transition_requests"] == 1


def test_bridge_saves_late_transition_callbacks_in_game_time_order(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    behavior_clone(_replay_data(), ACTION_DIM, torch.device("cpu"), checkpoint, epochs=1, batch_size=4)
    output = tmp_path / "rollout.npz"
    bridge = PolicyBridge(checkpoint, output, "cpu")
    newer = bridge.act({"observation": [0.0] * OBSERVATION_DIM, "action_mask": [True] + [False] * (ACTION_DIM - 1), "game_time": 2.0})
    older = bridge.act({"observation": [0.0] * OBSERVATION_DIM, "action_mask": [True] + [False] * (ACTION_DIM - 1), "game_time": 1.0})
    bridge.transition({"decision_id": newer["decision_id"], "reward": 1.0, "done": True})
    bridge.transition({"decision_id": older["decision_id"], "reward": 0.0, "done": False})
    bridge.flush()
    assert load_rollouts(output).game_times.tolist() == [1.0, 2.0]
    assert load_rollout_metadata(output)["transition_ordering"] == "game_time_sorted"


def test_paired_comparison_requires_exact_comparable_local_evidence(tmp_path: Path) -> None:
    def write_rollout(path: Path, *, reward: float, last_hit: bool) -> None:
        rewards = np.array([0.01, reward if last_hit else -0.01], np.float32)
        actions = np.array([0, ACTION_IDS["attack"] if last_hit else 0], np.int64)
        masks = np.zeros((2, ACTION_DIM), bool); masks[:, 0] = True; masks[1, ACTION_IDS["attack"]] = True
        rollout = Rollouts(np.zeros((2, OBSERVATION_DIM), np.float32), actions, rewards, np.array([False, True]),
                           "local_instrumented_lobby", np.zeros(2, np.float32), np.zeros(2, np.float32), masks,
                           np.array([0.0, 0.25], np.float32))
        save_rollouts(path, rollout, {"reward_version": CURRENT_LOCAL_REWARD_VERSION,
                                      "observation_version": CURRENT_LOCAL_OBSERVATION_VERSION,
                                      "policy_checkpoint_sha256": "same-policy"})
    baseline = [tmp_path / f"baseline_{index}.npz" for index in range(3)]
    candidate = [tmp_path / f"candidate_{index}.npz" for index in range(3)]
    for path in baseline: write_rollout(path, reward=0.0, last_hit=False)
    for path in candidate: write_rollout(path, reward=1.0, last_hit=True)
    result = compare_rollout_sets(baseline, candidate, minimum_game_seconds=0.2, output=tmp_path / "comparison.json")
    assert result["comparison"]["preliminary_improvement"] is True
    assert (tmp_path / "comparison.json").exists()


def test_bridge_writes_separate_calibration_events(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    behavior_clone(_replay_data(), ACTION_DIM, torch.device("cpu"), checkpoint, epochs=1, batch_size=4)
    calibration = tmp_path / "calibration.jsonl"
    bridge = PolicyBridge(checkpoint, tmp_path / "rollout.npz", "cpu", calibration_output=calibration)
    event = {"schema_version": SCHEMA_VERSION, "event": "transition", "decision_id": "d1", "action_name": "attack",
             "game_time": 12.0, "target_distance": 500.0, "attack_range": 500.0, "attack_damage": 70.0,
             "hero_move_speed": 300.0, "target_health": 65.0, "target_max_health": 550.0, "target_entindex": 42,
             "hero_health": 600.0, "hero_max_health": 600.0, "reward": 1.0, "enemy_count": 4.0,
             "ally_count": 4.0, "last_hit": True, "hero_dead": False, "done": True}
    assert bridge.record_calibration(event)["accepted"] is True
    no_target = {**event, "decision_id": "d2", "action_name": "move_east", "target_entindex": -1,
                 "target_health": 0.0, "target_max_health": 1.0, "game_time": 12.25, "last_hit": False}
    assert bridge.record_calibration(no_target)["accepted"] is True
    report = analyze_events([calibration])
    assert report["last_hits"] == 1
    assert report["targeted_events"] == 1
    assert report["measured_lane_config"]["hero_attack_range"] == 500.0
    assert bridge.health()["calibration_events"] == 2


def test_bridge_flush_commits_final_pending_decision(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    behavior_clone(_replay_data(), ACTION_DIM, torch.device("cpu"), checkpoint, epochs=1, batch_size=4)
    output = tmp_path / "rollout.npz"
    bridge = PolicyBridge(checkpoint, output, "cpu")
    bridge.act({"observation": [0.0] * OBSERVATION_DIM, "action_mask": [True] + [False] * (ACTION_DIM - 1)})
    result = bridge.flush({"final_reward": 0.75})
    rollout = load_rollouts(output)
    assert result["steps"] == 1
    assert rollout.dones.tolist() == [True]
    assert rollout.rewards.tolist() == [0.75]


def test_bridge_rotates_a_batch_filename_template(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    behavior_clone(_replay_data(), ACTION_DIM, torch.device("cpu"), checkpoint, epochs=1, batch_size=4)
    bridge = PolicyBridge(checkpoint, tmp_path / "lane_{batch:03d}.npz", "cpu")
    for reward in (0.25, 0.5):
        decision = bridge.act({"observation": [0.0] * OBSERVATION_DIM, "action_mask": [True] + [False] * (ACTION_DIM - 1)})
        bridge.transition({"decision_id": decision["decision_id"], "reward": reward, "done": True})
        bridge.flush()
    assert load_rollouts(tmp_path / "lane_001.npz").rewards.tolist() == [0.25]
    assert load_rollouts(tmp_path / "lane_002.npz").rewards.tolist() == [0.5]


def test_human_order_join_produces_bc_dataset(tmp_path: Path) -> None:
    match = tmp_path / "match"; match.mkdir()
    pd.DataFrame({"player_id": [1, 1], "tick": [0, 30], "x": [0, 100], "y": [0, 0]}).to_parquet(match / "positions.parquet")
    pd.DataFrame({"player_id": [1], "tick": [0], "net_worth": [600], "lh": [0], "xp": [0]}).to_parquet(match / "players_minute.parquet")
    source = tmp_path / "input.jsonl"
    source.write_text('{"tick":30,"player_id":1,"order":"use_item","inventory_slot":1}\n', encoding="utf-8")
    labels = tmp_path / "labels.parquet"; canonicalize_jsonl(source, labels)
    output = tmp_path / "human_bc.npz"
    result = join_orders_to_states(match, labels, output)
    data = load_trajectories(output)
    assert result["steps"] == 1
    assert data.actions[0] > 8  # item action, not a replay-inferred movement label


def test_merge_exact_rollouts(tmp_path: Path) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"rollout_{index}.npz"
        rollout = Rollouts(np.zeros((2, OBSERVATION_DIM), np.float32), np.array([0, 1]), np.ones(2, np.float32),
                           np.array([False, True]), "local_instrumented_lobby", np.zeros(2, np.float32),
                           np.zeros(2, np.float32), np.ones((2, ACTION_DIM), dtype=bool), np.array([index, index + 0.25], np.float32))
        save_rollouts(path, rollout, {"reward_version": CURRENT_LOCAL_REWARD_VERSION,
                                      "observation_version": CURRENT_LOCAL_OBSERVATION_VERSION,
                                      "policy_checkpoint_sha256": "same-policy"})
        paths.append(path)
    output = tmp_path / "merged.npz"
    result = merge_rollouts(paths, output)
    assert result["steps"] == 4
    assert load_rollouts(output).dones.sum() == 2
    assert load_rollouts(output).game_times is None
    assert load_rollout_metadata(output)["policy_checkpoint_sha256"] == "same-policy"


def test_merge_rejects_an_archive_from_another_reward_version(tmp_path: Path) -> None:
    path = tmp_path / "old_reward.npz"
    rollout = Rollouts(np.zeros((2, OBSERVATION_DIM), np.float32), np.array([0, 1]), np.ones(2, np.float32),
                       np.array([False, True]), "local_instrumented_lobby", np.zeros(2, np.float32),
                       np.zeros(2, np.float32), np.ones((2, ACTION_DIM), dtype=bool), np.array([0.0, 0.25], np.float32))
    save_rollouts(path, rollout, {"reward_version": "lane_wave_clear_v2"})
    try:
        merge_rollouts([path], tmp_path / "merged.npz")
    except ValueError as error:
        assert "reward_version" in str(error)
    else:
        raise AssertionError("merge accepted a different reward version")


def test_merge_rejects_an_archive_from_another_observation_version(tmp_path: Path) -> None:
    path = tmp_path / "old_observation.npz"
    rollout = Rollouts(np.zeros((2, OBSERVATION_DIM), np.float32), np.array([0, 1]), np.ones(2, np.float32),
                       np.array([False, True]), "local_instrumented_lobby", np.zeros(2, np.float32),
                       np.zeros(2, np.float32), np.ones((2, ACTION_DIM), dtype=bool), np.array([0.0, 0.25], np.float32))
    save_rollouts(path, rollout, {"reward_version": CURRENT_LOCAL_REWARD_VERSION,
                                  "observation_version": "lane_v2", "policy_checkpoint_sha256": "same-policy"})
    try:
        merge_rollouts([path], tmp_path / "merged.npz")
    except ValueError as error:
        assert "observation_version" in str(error)
    else:
        raise AssertionError("merge accepted a different observation contract")


def test_merge_rejects_a_rollout_that_ends_mid_episode(tmp_path: Path) -> None:
    path = tmp_path / "partial.npz"
    rollout = Rollouts(np.zeros((2, OBSERVATION_DIM), np.float32), np.array([0, 1]), np.ones(2, np.float32),
                       np.array([True, False]), "local_instrumented_lobby", np.zeros(2, np.float32),
                       np.zeros(2, np.float32), np.ones((2, ACTION_DIM), dtype=bool), np.array([0.0, 0.25], np.float32))
    save_rollouts(path, rollout, {"reward_version": CURRENT_LOCAL_REWARD_VERSION,
                                  "observation_version": CURRENT_LOCAL_OBSERVATION_VERSION,
                                  "policy_checkpoint_sha256": "same-policy"})
    try:
        merge_rollouts([path], tmp_path / "merged.npz")
    except ValueError as error:
        assert "mid-episode" in str(error)
    else:
        raise AssertionError("merge accepted a rollout that ends mid-episode")


def test_rollout_validation_rejects_invalid_mask_and_nonfinite_behavior_fields() -> None:
    bad_mask = Rollouts(np.zeros((1, OBSERVATION_DIM), np.float32), np.array([1]), np.zeros(1, np.float32),
                        np.array([True]), "local_instrumented_lobby", np.zeros(1, np.float32), np.zeros(1, np.float32),
                        np.array([[True] + [False] * (ACTION_DIM - 1)]), np.array([0.0], np.float32))
    try:
        bad_mask.validate(ACTION_DIM)
    except ValueError as error:
        assert "outside its action mask" in str(error)
    else:
        raise AssertionError("rollout accepted an action outside its mask")
    nonfinite = Rollouts(np.zeros((1, OBSERVATION_DIM), np.float32), np.array([0]), np.zeros(1, np.float32),
                         np.array([True]), "local_instrumented_lobby", np.array([np.nan], np.float32), np.zeros(1, np.float32),
                         np.ones((1, ACTION_DIM), dtype=bool), np.array([0.0], np.float32))
    try:
        nonfinite.validate(ACTION_DIM)
    except ValueError as error:
        assert "must be finite" in str(error)
    else:
        raise AssertionError("rollout accepted a non-finite behavior-policy field")


def test_evaluation_reports_lane_metrics(tmp_path: Path) -> None:
    path = tmp_path / "lane.npz"
    actions = np.array([0, 9, 0])
    masks = np.zeros((3, ACTION_DIM), dtype=bool); masks[:, 0] = True; masks[1, 9] = True
    rollout = Rollouts(np.zeros((3, OBSERVATION_DIM), np.float32), actions, np.array([0.0, 1.0, -2.0], np.float32),
                       np.array([False, True, True]), "local_instrumented_lobby", np.zeros(3, np.float32),
                       np.zeros(3, np.float32), masks, np.array([10.0, 10.25, 10.5], np.float32))
    save_rollouts(path, rollout, {})
    report = evaluate_rollouts([path])
    assert report["aggregate"]["completed_episodes"] == 2
    assert report["aggregate"]["attack_actions"] == 1
    assert report["aggregate"]["attack_allowed_steps"] == 1
    assert report["aggregate"]["last_hit_reward_signals"] == 1
    assert report["aggregate"]["death_reward_signals"] == 1
    assert report["aggregate"]["decisions_per_game_second"] == 6.0


def test_supervisor_rejects_low_cadence_rollout(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    behavior_clone(_replay_data(), ACTION_DIM, torch.device("cpu"), checkpoint, epochs=1, batch_size=4)
    path = tmp_path / "slow.npz"
    rollout = Rollouts(np.zeros((8, OBSERVATION_DIM), np.float32), np.zeros(8, np.int64), np.zeros(8, np.float32),
                       np.array([False] * 7 + [True]), "local_instrumented_lobby", np.zeros(8, np.float32),
                       np.zeros(8, np.float32), np.ones((8, ACTION_DIM), dtype=bool), np.arange(8, dtype=np.float32) * 2)
    save_rollouts(path, rollout, {"reward_version": CURRENT_LOCAL_REWARD_VERSION,
                                  "observation_version": CURRENT_LOCAL_OBSERVATION_VERSION,
                                  "policy_checkpoint_sha256": checkpoint_sha256(checkpoint)})
    config = SupervisorConfig(tmp_path, checkpoint, tmp_path / "run", min_steps=8, min_decisions_per_game_second=3,
                              min_game_seconds=0.2)
    accepted, result = validate_rollout(path, config)
    assert accepted is None
    assert "cadence" in str(result["reason"])


def test_supervisor_rejects_wrong_reward_version(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    behavior_clone(_replay_data(), ACTION_DIM, torch.device("cpu"), checkpoint, epochs=1, batch_size=4)
    path = tmp_path / "old_reward.npz"
    rollout = Rollouts(np.zeros((8, OBSERVATION_DIM), np.float32), np.zeros(8, np.int64), np.zeros(8, np.float32),
                       np.array([False] * 7 + [True]), "local_instrumented_lobby", np.zeros(8, np.float32),
                       np.zeros(8, np.float32), np.ones((8, ACTION_DIM), dtype=bool), np.arange(8, dtype=np.float32) * .25)
    save_rollouts(path, rollout, {"reward_version": "lane_wave_clear_v2"})
    config = SupervisorConfig(tmp_path, checkpoint, tmp_path / "run", min_steps=8, min_game_seconds=0.2)
    accepted, result = validate_rollout(path, config)
    assert accepted is None
    assert "reward_version" in str(result["reason"])


def test_supervisor_rejects_another_policy_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    behavior_clone(_replay_data(), ACTION_DIM, torch.device("cpu"), checkpoint, epochs=1, batch_size=4)
    path = tmp_path / "wrong_policy.npz"
    rollout = Rollouts(np.zeros((8, OBSERVATION_DIM), np.float32), np.zeros(8, np.int64), np.zeros(8, np.float32),
                       np.array([False] * 7 + [True]), "local_instrumented_lobby", np.zeros(8, np.float32),
                       np.zeros(8, np.float32), np.ones((8, ACTION_DIM), dtype=bool), np.arange(8, dtype=np.float32) * .25)
    save_rollouts(path, rollout, {"reward_version": CURRENT_LOCAL_REWARD_VERSION,
                                  "observation_version": CURRENT_LOCAL_OBSERVATION_VERSION,
                                  "policy_checkpoint_sha256": "not-the-configured-checkpoint"})
    config = SupervisorConfig(tmp_path, checkpoint, tmp_path / "run", min_steps=8, min_game_seconds=0.2)
    accepted, result = validate_rollout(path, config)
    assert accepted is None
    assert "configured checkpoint" in str(result["reason"])


def test_supervisor_evaluates_and_trains_one_valid_batch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    behavior_clone(_replay_data(), ACTION_DIM, torch.device("cpu"), checkpoint, epochs=1, batch_size=4)
    rollout_dir = tmp_path / "rollouts"; rollout_dir.mkdir()
    path = rollout_dir / "valid.npz"
    rollout = Rollouts(np.zeros((8, OBSERVATION_DIM), np.float32), np.zeros(8, np.int64), np.ones(8, np.float32),
                       np.array([False] * 7 + [True]), "local_instrumented_lobby", np.zeros(8, np.float32),
                       np.zeros(8, np.float32), np.ones((8, ACTION_DIM), dtype=bool), np.arange(8, dtype=np.float32) * 0.25)
    save_rollouts(path, rollout, {"reward_version": CURRENT_LOCAL_REWARD_VERSION,
                                  "observation_version": CURRENT_LOCAL_OBSERVATION_VERSION,
                                  "policy_checkpoint_sha256": checkpoint_sha256(checkpoint)})
    run_dir = tmp_path / "run"
    config = SupervisorConfig(rollout_dir, checkpoint, run_dir, batch_archives=1, max_batches=1, max_hours=1,
                              poll_seconds=0, min_archive_age_seconds=0, min_steps=8,
                              min_decisions_per_game_second=3, epochs=1, device="cpu", include_existing=True,
                              min_game_seconds=0.2)
    result = run_supervisor(config)
    assert result["completed_batches"] == 1
    assert Path(str(result["current_checkpoint"])).is_file()
    assert (run_dir / "evaluations" / "batch_001.json").is_file()
    assert "ppo_completed" in (run_dir / "audit.jsonl").read_text(encoding="utf-8")
