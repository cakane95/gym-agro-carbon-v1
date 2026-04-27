"""
Registration utilities for GAMA-backed contextual MDP environments.

Follows the same pattern as register.py:
    1. Factory returns a config dict
    2. registerContextualGamaMDP registers the env in Gymnasium
    3. make_gama calls the factory, registers, and returns the unwrapped env

Usage:
    from gama_register import make_gama

    env = make_gama(
        "gama-agrocarbon-agnostic",
        gaml_experiment_path="/usr/lib/gama/workspace/gama_models/EcoSysML/models/main.gaml",
    )
"""

import sys
import gymnasium
from gymnasium.envs.registration import register
import numpy as np

from src.contextual_stat_rl.environments.ContextualMDPs_discrete.factories.gama_agrocarbon_factory import (
    build_gama_agnostic_agrocarbon_config,
    build_gama_reward_contextual_agrocarbon_config,
)

INFINITY = sys.maxsize


# ==========================================
# Gymnasium Registration
# ==========================================

def registerContextualGamaMDP(
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
    # GAMA-specific kwargs
    gaml_experiment_path=None,
    gaml_experiment_name="gym_env",
    gama_ip_address="localhost",
    gama_port=6868,
    gaml_experiment_parameters=None,
):
    if name is None:
        name = f"ContextualGamaMDP-S{nS}_A{nA}_C{nC}_s{seed}-v0"

    register(
        id=name,
        entry_point="src.contextual_stat_rl.environments.ContextualMDPs_discrete.contextual_gama_env:ContextualGamaEnv",
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={
            # MDP kwargs (passed to ContextualDiscreteMDP parent)
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
            # GAMA kwargs (passed to ContextualGamaEnv)
            "gaml_experiment_path": gaml_experiment_path,
            "gaml_experiment_name": gaml_experiment_name,
            "gama_ip_address": gama_ip_address,
            "gama_port": gama_port,
            "gaml_experiment_parameters": gaml_experiment_parameters,
        },
    )
    return name


# ==========================================
# Registry
# ==========================================

registerContextualGamaRLenvironments = {
    "gama-agrocarbon-agnostic": build_gama_agnostic_agrocarbon_config,
    "gama-agrocarbon-reward-contextual": build_gama_reward_contextual_agrocarbon_config,
}


# ==========================================
# Utilities
# ==========================================

def print_gama_envlist():
    print("-" * 40)
    print("Registered GAMA contextual environments:")
    for k in registerContextualGamaRLenvironments:
        print(f"\t{k}")
    print("-" * 40)


def register_gama_env(envName, **override_kwargs):
    """Dynamically builds and registers the GAMA environment only when requested."""
    if envName in registerContextualGamaRLenvironments:
        # Call the factory with overrides to generate the config dictionary
        config = registerContextualGamaRLenvironments[envName](**override_kwargs)

        # Pass the unpacked config to the registration function
        regName = registerContextualGamaMDP(name=envName, **config)

        if not isinstance(regName, str):
            raise TypeError(f"Registered environment name must be a string, got {type(regName)}")
        print(f"[GAMA_REGISTER.INFO] Environment {envName} registered as {regName}")
        return regName
    else:
        return envName


def makeWorld(registername):
    return gymnasium.make(registername).unwrapped


def make_gama(envName, **override_kwargs):
    return makeWorld(register_gama_env(envName, **override_kwargs))