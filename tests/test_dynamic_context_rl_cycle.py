"""
Integration test: validates ContextualGamaEnv with dynamic contexts.

Tests:
    1. Environment creation via make_gama with c_is_static=False
    2. Reset returns correct (c, s) format
    3. Context can change within an episode
    4. State transition logic remains coherent under dynamic context
    5. Info contract after step
    6. Oracle access still works with local P/R
    7. Clean close

Usage (from host):
    docker-compose exec gym-agent python tests/test_dynamic_context_rl_cycle.py
"""

import os
import sys
import asyncio
import numpy as np


def log_test(name, status, detail=None):
    """Print a formatted test result."""
    line = f"[TEST] {name:<55} {status}"
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

    nS = 4
    nA = 4
    nC = 3
    trigger_action = 2
    steps_for_context_check = 40

    print("=" * 70)
    print("Dynamic Context Integration Test: GAMA + Contextual MDP")
    print(f"  GAML:                 {gaml_path}")
    print(f"  nS={nS}, nA={nA}, nC={nC}")
    print(f"  c_is_static:          False")
    print(f"  Context check steps:  {steps_for_context_check}")
    print("=" * 70)
    print()

    # -------------------------------------------------------
    # Test 1: Environment creation
    # -------------------------------------------------------
    try:
        from src.contextual_stat_rl.environments.gama_register import make_gama

        env = make_gama(
            "gama-agrocarbon-reward-contextual",
            nS=nS,
            nA=nA,
            nC=nC,
            trigger_action=trigger_action,
            p_cut=0.0,
            difficulty="easy",
            c_is_static=False,
            gaml_experiment_path=gaml_path,
        )

        assert env.c_is_static is False, (
            f"Expected env.c_is_static=False, got {env.c_is_static}"
        )

        log_test("1. Environment creation with dynamic context", "OK")
    except Exception as e:
        log_test("1. Environment creation with dynamic context", "FAILED", str(e))
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
        assert s == 0, f"Initial state should be 0, got {s}"
        assert "mean" in info, "Info should contain 'mean' key"

        log_test("2. Reset format (c, s)", "OK", f"obs={obs}")
    except Exception as e:
        log_test("2. Reset format (c, s)", "FAILED", str(e))
        env.close()
        sys.exit(1)

    # -------------------------------------------------------
    # Test 3: Context can change within an episode
    # -------------------------------------------------------
    try:
        obs, info = env.reset(seed=100)
        contexts = [obs[0]]
        states = [obs[1]]
        rewards = []

        for _ in range(steps_for_context_check):
            # Use action 0 to keep the tree state at 0.
            # This isolates context dynamics from tree-aging dynamics.
            obs, reward, done, truncated, info = env.step(0)
            c_t, s_t = obs

            assert 0 <= c_t < env.nC, f"Context {c_t} out of range"
            assert s_t == 0, f"State should remain 0 with action 0, got {s_t}"

            contexts.append(c_t)
            states.append(s_t)
            rewards.append(round(float(reward), 4))

        unique_contexts = sorted(set(contexts))

        assert len(unique_contexts) > 1, (
            "Dynamic context did not change during the episode. "
            f"Observed contexts: {contexts}"
        )

        log_test(
            "3. Dynamic context changes within episode",
            "OK",
            f"contexts_seen={unique_contexts}, first_sequence={contexts[:10]}"
        )
    except Exception as e:
        log_test("3. Dynamic context changes within episode", "FAILED", str(e))
        env.close()
        sys.exit(1)

    # -------------------------------------------------------
    # Test 4: Tree transition logic still works
    # -------------------------------------------------------
    try:
        obs, info = env.reset(seed=200)
        c0, s0 = obs
        assert s0 == 0, f"Initial state should be 0, got {s0}"

        # Initiate ANR/tree protection.
        obs, reward, done, truncated, info = env.step(trigger_action)
        c1, s1 = obs
        assert s1 == 1, f"After trigger action, state should be 1, got {s1}"

        # Tree ages automatically.
        obs, reward, done, truncated, info = env.step(0)
        c2, s2 = obs
        assert s2 == 2, f"After one aging step, state should be 2, got {s2}"

        obs, reward, done, truncated, info = env.step(0)
        c3, s3 = obs
        assert s3 == 3, f"After two aging steps, state should be 3, got {s3}"

        obs, reward, done, truncated, info = env.step(0)
        c4, s4 = obs
        assert s4 == env.nS - 1, (
            f"State should be capped at {env.nS - 1}, got {s4}"
        )

        log_test(
            "4. Transition logic under dynamic context",
            "OK",
            f"states: {s0}->{s1}->{s2}->{s3}->{s4}; "
            f"contexts: {c0}->{c1}->{c2}->{c3}->{c4}"
        )
    except Exception as e:
        log_test("4. Transition logic under dynamic context", "FAILED", str(e))
        env.close()
        sys.exit(1)

    # -------------------------------------------------------
    # Test 5: Info contract after step
    # -------------------------------------------------------
    try:
        obs, _ = env.reset(seed=250)
        obs, reward, done, truncated, info = env.step(trigger_action)

        assert "action_executed" in info, "Info missing 'action_executed'"
        assert "was_cut" in info, "Info missing 'was_cut'"
        assert "parcel_reward" in info, "Info missing 'parcel_reward'"
        assert "mean" in info, "Info missing 'mean'"

        assert info["action_executed"] == trigger_action, (
            f"action_executed should be {trigger_action}, "
            f"got {info['action_executed']}"
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

        log_test(
            "5. Info contract after dynamic-context step",
            "OK",
            f"obs={obs}, action_executed={info['action_executed']}, "
            f"was_cut={info['was_cut']}, "
            f"parcel_reward={info['parcel_reward']:.3f}, "
            f"mean={info['mean']:.3f}"
        )
    except Exception as e:
        log_test("5. Info contract after dynamic-context step", "FAILED", str(e))
        env.close()
        sys.exit(1)

    # -------------------------------------------------------
    # Test 6: Oracle access still works
    # -------------------------------------------------------
    try:
        obs, info = env.reset(seed=300)
        c, s = obs

        transition = env.getTransition(s, trigger_action, c=c)
        mean_reward = env.getMeanReward(s, trigger_action, c=c)

        assert isinstance(transition, np.ndarray), (
            f"Expected ndarray, got {type(transition)}"
        )
        assert len(transition) == env.nS, (
            f"Transition length {len(transition)} != nS={env.nS}"
        )
        assert abs(sum(transition) - 1.0) < 1e-6, (
            f"Transition does not sum to 1: {sum(transition)}"
        )
        assert isinstance(mean_reward, float), (
            f"Expected float, got {type(mean_reward)}"
        )

        log_test(
            "6. Oracle access under dynamic context",
            "OK",
            f"obs={obs}, P(s={s}, a={trigger_action}, c={c})={transition}, "
            f"mean_r={mean_reward:.3f}"
        )
    except Exception as e:
        log_test("6. Oracle access under dynamic context", "FAILED", str(e))
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

    print()
    print("=" * 70)
    print("[TEST] All dynamic-context integration tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())