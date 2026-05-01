"""
YAML-driven experiment runner for GAMA-backed contextual MDP environments.

Reads a scenario YAML config, instantiates the environment and agents,
builds the oracle on a local pure-Python env, and runs the full experiment
(regret + action distribution plots).

Usage:
    docker-compose exec gym-agent python articles/neurips26/scripts/run_experiment.py \
        articles/neurips26/configs/scenario_1_easy_det.yaml
"""

import os
import sys
import asyncio
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.contextual_stat_rl.environments.gama_register import make_gama
from src.contextual_stat_rl.experiments.sequential_experiment import (
    runSequentialGamaExperiment,
    build_oracle_env,
)
from src.contextual_stat_rl.learners.ContextualMDPs_discrete.Optimal import (
    ContextualOptimalControl as opt,
)

# Agent class registry: maps YAML class names to actual classes
from src.contextual_stat_rl.learners.ContextualMDPs_discrete.ETC import GlobalETC
from src.contextual_stat_rl.learners.ContextualMDPs_discrete.ContextualIMED_RL import (
    GlobalIMEDRL,
    SemiLocalIMEDRL,
)

AGENT_REGISTRY = {
    "GlobalETC": GlobalETC,
    "GlobalIMEDRL": GlobalIMEDRL,
    "SemiLocalIMEDRL": SemiLocalIMEDRL,
}


def load_config(config_path):
    """Load and return the YAML config as a dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_agents(agent_configs, nS, nA, nC, skeleton):
    """
    Build the agent definition list from YAML config.

    Each agent config has:
        class: str (key in AGENT_REGISTRY)
        params: dict (agent-specific parameters)

    Environment-derived params (nS, nA, nC, skeleton) are injected
    automatically based on the agent class signature.
    """
    agents = []

    for agent_cfg in agent_configs:
        class_name = agent_cfg["class"]
        if class_name not in AGENT_REGISTRY:
            raise ValueError(
                f"Unknown agent class: '{class_name}'. "
                f"Available: {list(AGENT_REGISTRY.keys())}"
            )

        agent_class = AGENT_REGISTRY[class_name]
        params = dict(agent_cfg.get("params", {}))

        # Inject environment params based on agent type
        if class_name == "GlobalETC":
            params["nS"] = nS
            params["nA"] = nA
            params["nC"] = nC
            params["skeleton"] = skeleton
        elif class_name in ("GlobalIMEDRL", "SemiLocalIMEDRL"):
            params["nbr_states"] = nS
            params["nbr_actions"] = nA
            params["nbr_contexts"] = nC
            params["skeleton"] = skeleton

        agents.append((agent_class, params))

    return agents


async def main():
    # -------------------------------------------------------
    # Parse command line
    # -------------------------------------------------------
    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <config.yaml>")
        print("Example: python run_experiment.py configs/scenario_1_easy_det.yaml")
        sys.exit(1)

    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # -------------------------------------------------------
    # Extract config sections
    # -------------------------------------------------------
    exp_cfg = config["experiment"]
    env_cfg = config["environment"]
    gama_cfg = config.get("gama", {})
    farmer_cfg = gama_cfg.get("farmer", {})
    agent_cfgs = config["agents"]

    rel_config_path = os.path.relpath(config_path, os.path.join(SCRIPT_DIR, "..", "configs"))
    scenario_name = os.path.splitext(rel_config_path)[0].replace(os.sep, "__")

    print("=" * 60)
    print(f"GymAgroCarbon Experiment: {scenario_name}")
    print(f"  Environment:  {exp_cfg['name']}")
    print(f"  nS={env_cfg['nS']}  nA={env_cfg['nA']}  nC={env_cfg['nC']}")
    print(f"  Difficulty:   {env_cfg['difficulty']}")
    print(f"  p_cut:        {env_cfg['p_cut']}")
    print(f"  Horizon:      {exp_cfg['timeHorizon']}")
    print(f"  Replicates:   {exp_cfg['nbReplicates']}")
    print(f"  Agents:       {[a['class'] for a in agent_cfgs]}")
    print("=" * 60)

    # -------------------------------------------------------
    # Resolve GAMA connection
    # -------------------------------------------------------
    gaml_path = gama_cfg.get(
        "gaml_experiment_path",
        os.environ.get(
            "GAML_PATH",
            "/usr/lib/gama/workspace/gama_models/EcoSysML/models/main.gaml"
        ),
    )
    gaml_experiment_name = gama_cfg.get("gaml_experiment_name", "gym_env")

    # -------------------------------------------------------
    # Create GAMA environment
    # -------------------------------------------------------
    env = make_gama(
        exp_cfg["name"],
        nS=env_cfg["nS"],
        nA=env_cfg["nA"],
        nC=env_cfg["nC"],
        trigger_action=env_cfg.get("trigger_action", 2),
        p_cut=env_cfg["p_cut"],
        difficulty=env_cfg["difficulty"],
        compliance_params=farmer_cfg,
        gaml_experiment_path=gaml_path,
        gaml_experiment_name=gaml_experiment_name,
    )

    nS = env.nS
    nA = env.nA
    nC = env.nC
    skeleton = env.skeleton

    # -------------------------------------------------------
    # Build agents from YAML
    # -------------------------------------------------------
    agents = build_agents(agent_cfgs, nS, nA, nC, skeleton)

    # -------------------------------------------------------
    # Build oracle on pure-Python env
    # -------------------------------------------------------
    oracle_env = build_oracle_env(env)
    oracle = opt.build_opti(oracle_env.name, oracle_env, nS, nA)

    # -------------------------------------------------------
    # Setup output directory
    # -------------------------------------------------------
    results_dir = os.path.join(SCRIPT_DIR, "..", "results", scenario_name) + os.sep
    os.makedirs(results_dir, exist_ok=True)

    # -------------------------------------------------------
    # Run experiment
    # -------------------------------------------------------
    print(f"\nResults will be saved to: {results_dir}")
    print(f"Starting experiment...\n")

    try:
        runSequentialGamaExperiment(
            env=env,
            agents=agents,
            oracle=oracle,
            timeHorizon=exp_cfg["timeHorizon"],
            opttimeHorizon=exp_cfg.get("opttimeHorizon", 10000),
            nbReplicates=exp_cfg["nbReplicates"],
            root_folder=results_dir,
            oracle_env=oracle_env,
        )
        print(f"\n[DONE] Experiment '{scenario_name}' completed successfully.")

    finally:
        env.close()


if __name__ == "__main__":
    asyncio.run(main())