import os
import sys
from statisticalrl_experiments.fullExperiment import runLargeMulticoreExperiment as xp

# 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

# 2. Package Imports
from src.contextual_stat_rl.learners.ContextualMDPs_discrete.ETC import GlobalETC4
from src.contextual_stat_rl.learners.ContextualMDPs_discrete.ContextualIMED_RL import GlobalIMEDRL, SemiLocalIMEDRL
from src.contextual_stat_rl.learners.ContextualMDPs_discrete.Optimal import ContextualOptimalControl as opt
import src.contextual_stat_rl.environments.register as cW


if __name__ == "__main__":
    # Instantiate the environment using the Factory
    env = cW.make("basic-agrocarbon-context")

    
    nS = env.unwrapped.nS
    nA = env.unwrapped.nA
    nC = env.unwrapped.nC
    skeleton = env.unwrapped.skeleton

    # 5. Define the Agents
    agents = [
        (
            GlobalETC4,
            {
                "nS": nS,
                "nA": nA,
                "nC": nC,
                "skeleton": skeleton,
                "gamma": 0.99,
                "epsilon": 1e-6,
                "max_iter": 1000,
                "name": "GlobalETC4",
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
                "max_reward": 1.5,
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
                "max_reward": 1.5,
            },
        ),
    ]

    # 6. Build the Oracle using the updated Contextual adapter
    oracle = opt.build_opti(env.unwrapped.name, env, nS, nA)

    # 7. Setup output directory and execute
    root_folder = os.path.join(CURRENT_DIR, "results") + os.sep
    
    # Ensure the results directory exists
    os.makedirs(root_folder, exist_ok=True)

    print(f"Starting experiment on {env.unwrapped.name}...")
    
    xp(
        env,
        agents,
        oracle,
        timeHorizon=20,
        nbReplicates=10,
        root_folder=root_folder,
    )