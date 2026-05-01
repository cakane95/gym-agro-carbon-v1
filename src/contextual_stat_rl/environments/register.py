import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import sys


from src.contextual_stat_rl.environments.ContextualMDPs_discrete.factories.agrocarbon_factory import (
    build_agnostic_agrocarbon_config,
    build_reward_contextual_agrocarbon_config,
)

INFINITY = sys.maxsize

def registerContextualMDP(
    nS,
    nA,
    P,
    R,
    mu0,
    nC,
    nu,
    skeleton=None,
    c_is_static=True,
    p_is_contextual=True,
    r_is_contextual=False,
    nameActions=None,
    seed=None,
    max_steps=INFINITY,
    reward_threshold=np.inf,
    name=None,
):
    if name is None:
        name = f"ContextualMDP-S{nS}_A{nA}_C{nC}_s{seed}-v0"

    register(
        id=name,
        entry_point="src.contextual_stat_rl.environments.ContextualMDPs_discrete.contextualMDP:ContextualDiscreteMDP",
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={
            "nS": nS,
            "nA": nA,
            "P": P,
            "R": R,
            "mu0": mu0,
            "nC": nC,
            "nu": nu,
            "skeleton": skeleton,
            "c_is_static": c_is_static,
            "p_is_contextual": p_is_contextual,
            "r_is_contextual": r_is_contextual,
            "nameActions": nameActions,
            "seed": seed,
            "name": name,
        },
    )
    return name


# 2. The Clean Registry Dictionary
registerContextualStatisticalRLenvironments = {
    "basic-agrocarbon-context": build_agnostic_agrocarbon_config,
    "agrocarbon-agnostic": build_agnostic_agrocarbon_config,
    "agrocarbon-reward-contextual": build_reward_contextual_agrocarbon_config,
}


def print_envlist():
    print("-" * 30)
    print("List of registered contextual environments:")
    for k in registerContextualStatisticalRLenvironments.keys():
        print("\t" + k)
    print("-" * 30)


def register_env(envName, **kwargs):
    """Dynamically builds and registers the environment only when requested."""
    if envName in registerContextualStatisticalRLenvironments.keys():
        config = registerContextualStatisticalRLenvironments[envName](**kwargs)

        regName = registerContextualMDP(name=envName, **config)

        if not isinstance(regName, str):
            raise TypeError(f"Registered environment name must be a string, got {type(regName)}")
        print(f"[REGISTER.INFO] Environment {envName} registered as {regName}")
        return regName
    else:
        return envName


def makeWorld(registername):
    return gymnasium.make(registername).unwrapped


def make(envName, **kwargs):
    return makeWorld(register_env(envName, **kwargs))