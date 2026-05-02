"""
Minimal experiment test in contextual_reward env.

Usage:
    docker-compose exec gym-agent python tests/test_mini_experiment.py
"""

import os
import sys
import asyncio

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.contextual_stat_rl.experiments.sequential_experiment import (
    runSequentialGamaExperiment as xp,
    build_oracle_env,
)

from src.contextual_stat_rl.learners.ContextualMDPs_discrete.ETC import GlobalETC
from src.contextual_stat_rl.learners.ContextualMDPs_discrete.ContextualIMED_RL import GlobalIMEDRL, SemiLocalIMEDRL
from src.contextual_stat_rl.learners.ContextualMDPs_discrete.ContextualUCRL3 import GlobalUCRL3
from src.contextual_stat_rl.learners.ContextualMDPs_discrete.ContextualQLearning import GlobalQLearning
from src.contextual_stat_rl.learners.ContextualMDPs_discrete.Optimal import ContextualOptimalControl as opt
from src.contextual_stat_rl.environments.gama_register import make_gama


async def main():
    gaml_path = os.environ.get(
        "GAML_PATH",
        "/usr/lib/gama/workspace/gama_models/EcoSysML/models/main.gaml"
    )

    print("=" * 60)
    print("Mini Contextual Reward Experiment: GAMA + RL Agents")
    print(f"  GAML:         {gaml_path}")
    print(f"  Horizon:      20")
    print(f"  Replicates:   10")
    print("=" * 60)

    # 1. Register and create the environment
    #    register_gama_env returns the registered gym name
    #    make_gama creates the first instance (for oracle + setup)
    env = make_gama(
        "gama-agrocarbon-reward-contextual",
        gaml_experiment_path=gaml_path,
    )

    nS = env.nS
    nA = env.nA
    nC = env.nC
    skeleton = env.skeleton

    # 2. Define agents (same as run_basic_agrocarbon.py)
    agents = [
        (
            GlobalETC,
            {
                "nS": nS,
                "nA": nA,
                "nC": nC,
                "skeleton": skeleton,
                "gamma": 0.99,
                "epsilon": 1e-6,
                "max_iter": 1000,
                "exploration_limit":10,
            },
        ),
        (
            GlobalIMEDRL,
            {
                "nbr_states": nS,
                "nbr_actions": nA,
                "nbr_contexts": nC,
                "skeleton": skeleton,
                "max_iter": 3000,
                "epsilon": 1e-3,
                "max_reward": 2.5,
            },
        ),
        (
            SemiLocalIMEDRL,
            {
                "nbr_states": nS,
                "nbr_actions": nA,
                "nbr_contexts": nC,
                "skeleton": skeleton,
                "max_iter": 3000,
                "epsilon": 1e-3,
                "max_reward": 2.5,
            },
        ),
        (
            GlobalUCRL3,
            {
                "nS": nS,
                "nA": nA,
                "nC": nC,
                "delta": 0.05,
                "K": -1,
                "max_reward": 2.5,
                "name": "GlobalUCRL3",
            },
        ),
        (
            GlobalQLearning,
            {
                "nS": nS,
                "nA": nA,
                "nC": nC,
                "gamma": 0.99,
                "epsilon": 0.3,
                "epsilon_min": 0.02,
                "epsilon_decay": 0.98,
                "alpha": 0.1,
                "optimistic_init": 0.0,
                "name": "GlobalQLearning",
            },
        ),
    ]

    # 3. Build oracle on pure-Python env (no GAMA overhead)
    oracle_env = build_oracle_env(env)
    oracle = opt.build_opti(oracle_env.name, oracle_env, nS, nA)

    # 4. Setup output
    root_folder = os.path.join(CURRENT_DIR, "results_gama") + os.sep
    os.makedirs(root_folder, exist_ok=True)

    # 5. Run experiment (small scale)
    print(f"\nStarting experiment on {env.name}...")

    try:
        xp(
            env,
            agents,
            oracle,
            timeHorizon=20,
            nbReplicates=10,
            root_folder=root_folder,
            oracle_env=oracle_env,
        )

        print("\n[DONE] Mini experiment completed.")

    # 6. Cleanup the initial env
    finally:
        env.close()


if __name__ == "__main__":
    asyncio.run(main())