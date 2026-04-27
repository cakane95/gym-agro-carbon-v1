"""
Integration test: validates ContextualGamaEnv full RL cycle.

Tests:
    1. Environment creation via make_gama
    2. Reset returns correct (c, s) format
    2b. Reset is not a step (reward=0, no simulation_cycle)
    3. Multiple steps with different actions
    3b. Info contract (action_executed, was_cut, parcel_reward present)
    4. Transition logic coherence (trigger_action plants tree, aging works)
    5. Multiple episodes (reset between episodes)
    6. Oracle access (getTransition, getMeanReward still work on local P/R)
    7. Clean close

Usage (from inside the gym-agent container):
    python test_rl_cycle.py

Or from host:
    docker-compose exec gym-agent python test_rl_cycle.py
"""

import os
import sys
import asyncio
import numpy as np


def log_test(name, status, detail=None):
    """Print a formatted test result."""
    line = f"[TEST] {name:<50} {status}"
    print(line)
    if detail:
        print(f"       {detail}")


async def main():
    # -------------------------------------------------------
    # Configuration
    # -------------------------------------------------------
    gaml_path = os.environ.get(
        "GAML_PATH",
        "/usr/lib/gama/workspace/gama_models/EcoSysML/models/main.gaml"
    )

    n_episodes = 3
    steps_per_episode = 10
    trigger_action = 2

    print("=" * 60)
    print("ContextualGamaEnv Integration Test")
    print(f"  GAML:             {gaml_path}")
    print(f"  Episodes:         {n_episodes}")
    print(f"  Steps/episode:    {steps_per_episode}")
    print(f"  Trigger action:   {trigger_action}")
    print("=" * 60)
    print()

    # -------------------------------------------------------
    # Test 1: Environment creation
    # -------------------------------------------------------
    try:
        from src.contextual_stat_rl.environments.gama_register import make_gama

        env = make_gama(
            "gama-agrocarbon-agnostic",
            gaml_experiment_path=gaml_path,
        )
        log_test("1. Environment creation", "OK")
    except Exception as e:
        log_test("1. Environment creation", "FAILED", str(e))
        sys.exit(1)

    # -------------------------------------------------------
    # Test 2: Reset returns correct format
    # -------------------------------------------------------
    try:
        obs, info = env.reset(seed=42)
        c, s = obs

        assert isinstance(obs, tuple), f"Expected tuple, got {type(obs)}"
        assert len(obs) == 2, f"Expected length 2, got {len(obs)}"
        assert 0 <= c < env.nC, f"Context {c} out of range [0, {env.nC})"
        assert 0 <= s < env.nS, f"State {s} out of range [0, {env.nS})"
        assert "mean" in info, "Info should contain 'mean' key"

        log_test("2. Reset format (c, s)", "OK", f"c={c}, s={s}")
    except Exception as e:
        log_test("2. Reset format (c, s)", "FAILED", str(e))
        env.close()
        sys.exit(1)

    # -------------------------------------------------------
    # Test 2b: Reset is not a step
    # -------------------------------------------------------
    try:
        obs, info = env.reset(seed=42)

        # Read gym_interface.data directly to check GAMA's state after reset
        reset_data = env.gama_client._execute_expression(
            env.experiment_id,
            r"gym_interface.data"
        )

        reset_reward = float(reset_data["Reward"])
        reset_message = reset_data["Info"].get("message", "")

        assert reset_reward == 0.0, (
            f"Reset should produce reward=0.0, got {reset_reward}"
        )
        assert reset_message == "reset_from_python", (
            f"Reset message should be 'reset_from_python', got '{reset_message}'"
        )

        log_test("2b. Reset is not a step", "OK",
                 f"reward={reset_reward}, message='{reset_message}'")
    except Exception as e:
        log_test("2b. Reset is not a step", "FAILED", str(e))
        env.close()
        sys.exit(1)

    # -------------------------------------------------------
    # Test 3: Multiple steps with action 0 (no tree)
    # -------------------------------------------------------
    try:
        obs, info = env.reset(seed=100)
        c0, s0 = obs

        for t in range(3):
            obs, reward, done, truncated, info = env.step(0)  # fallow
            c_t, s_t = obs

            assert c_t == c0, f"Context changed mid-episode: {c0} -> {c_t}"
            assert s_t == 0, f"State should stay 0 without planting, got {s_t}"
            assert isinstance(reward, float), f"Reward should be float, got {type(reward)}"
            assert "mean" in info, "Info should contain 'mean' key"

        log_test("3. Steps with action 0 (no planting)", "OK",
                 f"State stayed at 0, rewards received")
    except Exception as e:
        log_test("3. Steps with action 0 (no planting)", "FAILED", str(e))
        env.close()
        sys.exit(1)

    # -------------------------------------------------------
    # Test 3b: Info contract after step
    # -------------------------------------------------------
    try:
        obs, _ = env.reset(seed=150)
        obs, reward, done, truncated, info = env.step(trigger_action)

        assert "action_executed" in info, "Info missing 'action_executed'"
        assert "was_cut" in info, "Info missing 'was_cut'"
        assert "parcel_reward" in info, "Info missing 'parcel_reward'"
        assert "mean" in info, "Info missing 'mean'"

        assert info["action_executed"] == trigger_action, (
            f"action_executed should be {trigger_action}, got {info['action_executed']}"
        )
        assert isinstance(info["was_cut"], bool), (
            f"was_cut should be bool, got {type(info['was_cut'])}"
        )
        assert isinstance(info["parcel_reward"], float), (
            f"parcel_reward should be float, got {type(info['parcel_reward'])}"
        )
        assert isinstance(info["mean"], float), (
            f"mean should be float, got {type(info['mean'])}"
        )

        log_test("3b. Info contract after step", "OK",
                 f"action_executed={info['action_executed']}, "
                 f"was_cut={info['was_cut']}, "
                 f"parcel_reward={info['parcel_reward']:.3f}, "
                 f"mean={info['mean']:.3f}")
    except Exception as e:
        log_test("3b. Info contract after step", "FAILED", str(e))
        env.close()
        sys.exit(1)

    # -------------------------------------------------------
    # Test 4: Trigger action plants tree, aging works
    # -------------------------------------------------------
    try:
        obs, info = env.reset(seed=200)
        c0, s0 = obs
        assert s0 == 0, f"Initial state should be 0, got {s0}"

        # Plant the tree
        obs, reward, done, truncated, info = env.step(trigger_action)
        c1, s1 = obs
        assert s1 == 1, f"After planting, state should be 1, got {s1}"

        # Tree ages automatically
        obs, reward, done, truncated, info = env.step(0)  # any action
        c2, s2 = obs
        assert s2 == 2, f"After aging, state should be 2, got {s2}"

        # Ages again
        obs, reward, done, truncated, info = env.step(0)
        c3, s3 = obs
        assert s3 == 3, f"After aging, state should be 3, got {s3}"

        # Capped at nS-1
        obs, reward, done, truncated, info = env.step(0)
        c4, s4 = obs
        assert s4 == 3, f"State should be capped at {env.nS - 1}, got {s4}"

        log_test("4. Transition logic (plant + aging + cap)", "OK",
                 f"Trajectory: {s0} -> {s1} -> {s2} -> {s3} -> {s4}")
    except Exception as e:
        log_test("4. Transition logic (plant + aging + cap)", "FAILED", str(e))
        env.close()
        sys.exit(1)

    # -------------------------------------------------------
    # Test 5: Strict seeding check
    # -------------------------------------------------------
    try:
        def collect_rewards(seed):
            obs, info = env.reset(seed=seed)
            rewards = []

            for t in range(steps_per_episode):
                action = trigger_action if t == 0 else 0
                obs, reward, done, truncated, info = env.step(action)
                rewards.append(round(float(reward), 6))

            return rewards

        rewards_300_a = collect_rewards(300)
        rewards_301 = collect_rewards(301)
        rewards_300_b = collect_rewards(300)

        print(f"       seed=300 | run A | rewards={rewards_300_a}")
        print(f"       seed=301 | run B | rewards={rewards_301}")
        print(f"       seed=300 | run C | rewards={rewards_300_b}")

        assert rewards_300_a == rewards_300_b, (
            "Same seed should reproduce the same GAMA reward sequence:\n"
            f"seed=300 run A: {rewards_300_a}\n"
            f"seed=300 run C: {rewards_300_b}"
        )

        assert rewards_300_a != rewards_301, (
            "Different seeds should produce different GAMA reward sequences:\n"
            f"seed=300: {rewards_300_a}\n"
            f"seed=301: {rewards_301}"
        )

        log_test(
            "5. Strict seeding check",
            "OK",
            "Same seed reproduces same sequence; different seeds produce different sequences."
        )

    except Exception as e:
        log_test("5. Strict seeding check", "FAILED", str(e))
        env.close()
        sys.exit(1)

    # -------------------------------------------------------
    # Test 6: Oracle access (local P/R)
    # -------------------------------------------------------
    try:
        transition = env.getTransition(0, trigger_action)
        mean_reward = env.getMeanReward(0, trigger_action)

        assert isinstance(transition, np.ndarray), f"Expected ndarray, got {type(transition)}"
        assert len(transition) == env.nS, f"Transition length {len(transition)} != nS={env.nS}"
        assert abs(sum(transition) - 1.0) < 1e-6, f"Transition doesn't sum to 1: {sum(transition)}"
        assert isinstance(mean_reward, float), f"Expected float, got {type(mean_reward)}"

        log_test("6. Oracle access (getTransition, getMeanReward)", "OK",
                 f"P(s=0, a={trigger_action}) = {transition}, mean_r = {mean_reward:.3f}")
    except Exception as e:
        log_test("6. Oracle access (getTransition, getMeanReward)", "FAILED", str(e))
        env.close()
        sys.exit(1)

    # -------------------------------------------------------
    # Test 7: Clean close
    # -------------------------------------------------------
    try:
        env.close()
        assert env.gama_client is None, "gama_client should be None after close"
        log_test("7. Clean close", "OK")
    except Exception as e:
        log_test("7. Clean close", "FAILED", str(e))
        sys.exit(1)

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    print()
    print("=" * 60)
    print("[TEST] All 9 tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())