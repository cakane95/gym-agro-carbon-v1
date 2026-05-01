"""
Agent integration test: verifies that RL agents can interface with ContextualGamaEnv.

Tests that each agent type can:
    1. Be instantiated with the environment
    2. Receive a reset observation (c, s)
    3. Play actions via play(observation)
    4. Update via update(obs, action, reward, next_obs)
    5. Run a full mini-episode without crashing

No regret computation, no multiple runs — just branchement validation.

Usage:
    docker-compose exec gym-agent python tests/test_agent_branchement.py
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
    n_steps = 10

    print("=" * 60)
    print("Learner Link Test")
    print(f"  GAML:     {gaml_path}")
    print(f"  Steps:    {n_steps}")
    print("=" * 60)
    print()

    # -------------------------------------------------------
    # Imports
    # -------------------------------------------------------
    from src.contextual_stat_rl.environments.gama_register import make_gama
    from src.contextual_stat_rl.learners.ContextualMDPs_discrete.Optimal.ContextualOptimalControl import (
        GlobalOpti_controller,
    )
    from src.contextual_stat_rl.learners.ContextualMDPs_discrete.ETC import GlobalETC
    from src.contextual_stat_rl.learners.ContextualMDPs_discrete.ContextualIMED_RL import (
        GlobalIMEDRL,
        SemiLocalIMEDRL,
    )

    # -------------------------------------------------------
    # Create environment
    # -------------------------------------------------------
    try:
        env = make_gama(
            "gama-agrocarbon-agnostic",
            gaml_experiment_path=gaml_path,
        )
        nS = env.nS
        nA = env.nA
        nC = env.nC
        skeleton = env.skeleton

        log_test("0. Environment creation", "OK")
    except Exception as e:
        log_test("0. Environment creation", "FAILED", str(e))
        sys.exit(1)

    # -------------------------------------------------------
    # Test 1: Oracle (GlobalOpti_controller)
    # -------------------------------------------------------
    try:
        oracle = GlobalOpti_controller(env, nS, nA)

        obs, info = env.reset(seed=42)
        oracle.reset(obs)

        rewards = []
        for t in range(n_steps):
            action = oracle.play(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            oracle.update(obs, action, reward, next_obs)
            rewards.append(round(float(reward), 3))
            obs = next_obs

        log_test("1. Oracle (GlobalOpti_controller)", "OK",
                 f"rewards={rewards[:5]}...")
    except Exception as e:
        log_test("1. Oracle (GlobalOpti_controller)", "FAILED", str(e))
        env.close()
        sys.exit(1)

    # -------------------------------------------------------
    # Test 2: GlobalETC
    # -------------------------------------------------------
    try:
        etc_agent = GlobalETC(
            nS=nS,
            nA=nA,
            nC=nC,
            skeleton=skeleton,
            gamma=0.99,
            epsilon=1e-6,
            max_iter=1000,
            exploration_limit=10,
            name="GlobalETC3",
        )

        obs, info = env.reset(seed=100)
        etc_agent.reset(obs)

        rewards = []
        for t in range(n_steps):
            action = etc_agent.play(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            etc_agent.update(obs, action, reward, next_obs)
            rewards.append(round(float(reward), 3))
            obs = next_obs

        log_test("2. GlobalETC", "OK",
                 f"rewards={rewards[:5]}...")
    except Exception as e:
        log_test("2. GlobalETC4", "FAILED", str(e))
        env.close()
        sys.exit(1)

    # -------------------------------------------------------
    # Test 3: GlobalIMED-RL
    # -------------------------------------------------------
    try:
        imed_global = GlobalIMEDRL(
            nbr_states=nS,
            nbr_actions=nA,
            nbr_contexts=nC,
            skeleton=skeleton,
            max_iter=3000,
            epsilon=1e-3,
            max_reward=1.5,
        )

        obs, info = env.reset(seed=200)
        imed_global.reset(obs)

        rewards = []
        for t in range(n_steps):
            action = imed_global.play(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            imed_global.update(obs, action, reward, next_obs)
            rewards.append(round(float(reward), 3))
            obs = next_obs

        log_test("3. GlobalIMED-RL", "OK",
                 f"rewards={rewards[:5]}...")
    except Exception as e:
        log_test("3. GlobalIMED-RL", "FAILED", str(e))
        env.close()
        sys.exit(1)

    # -------------------------------------------------------
    # Test 4: SemiLocalIMED-RL
    # -------------------------------------------------------
    try:
        imed_semilocal = SemiLocalIMEDRL(
            nbr_states=nS,
            nbr_actions=nA,
            nbr_contexts=nC,
            skeleton=skeleton,
            max_iter=3000,
            epsilon=1e-3,
            max_reward=1.5,
        )

        obs, info = env.reset(seed=300)
        imed_semilocal.reset(obs)

        rewards = []
        for t in range(n_steps):
            action = imed_semilocal.play(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            imed_semilocal.update(obs, action, reward, next_obs)
            rewards.append(round(float(reward), 3))
            obs = next_obs

        log_test("4. SemiLocalIMED-RL", "OK",
                 f"rewards={rewards[:5]}...")
    except Exception as e:
        log_test("4. SemiLocalIMED-RL", "FAILED", str(e))
        env.close()
        sys.exit(1)

    # -------------------------------------------------------
    # Test 5: All agents produce valid actions from same obs
    # -------------------------------------------------------
    try:
        obs, info = env.reset(seed=400)

        agents = {
            "Oracle": oracle,
            "GlobalETC": etc_agent,
            "GlobalIMED-RL": imed_global,
            "SemiLocalIMED-RL": imed_semilocal,
        }

        for name, agent in agents.items():
            agent.reset(obs)
            action = agent.play(obs)
            assert isinstance(action, (int, np.integer)), (
                f"{name}.play() returned {type(action)}, expected int"
            )
            assert 0 <= action < nA, (
                f"{name}.play() returned action {action} out of range [0, {nA})"
            )

        log_test("5. All agents produce valid actions", "OK",
                 f"obs={obs}, all actions in [0, {nA})")
    except Exception as e:
        log_test("5. All agents produce valid actions", "FAILED", str(e))
        env.close()
        sys.exit(1)

    env.close()

    print()
    print("=" * 60)
    print("[TEST] All learner agent tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    import numpy as np
    asyncio.run(main())