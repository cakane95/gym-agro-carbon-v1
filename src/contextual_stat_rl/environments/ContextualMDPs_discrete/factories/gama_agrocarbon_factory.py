"""
Factory functions for GAMA-backed agrocarbon contextual MDP environments.

These return the same MDP kwargs as the pure-Python factories
(from agrocarbon_factory.py), plus GAMA connection parameters.
"""

import os

from src.contextual_stat_rl.environments.ContextualMDPs_discrete.factories.agrocarbon_factory import (
    build_tree_transitions,
    build_agnostic_reward_matrix,
    build_contextual_reward_matrix,
    build_initial_state_dist,
    build_context_dist,
    _build_context_scales,
    build_tree_skeleton,
    build_action_names,
    _build_action_bonus_scales,
    _build_base_means
)

def _build_compliance_params(
    household_size=1,
    tree_knowledge=1.0,
    base_compliance=1.0,
    food_pressure_penalty=0.0,
    tree_knowledge_bonus=0.0,
    fallback_action=3,
):
    """
    Build GAMA parameters controlling the Farmer BDI compliance behavior.
    """
    return [
        {"name": "Farmer Household Size", "type": "int", "value": household_size},
        {"name": "Farmer Tree Knowledge", "type": "float", "value": tree_knowledge},
        {"name": "Farmer Base Compliance", "type": "float", "value": base_compliance},
        {"name": "Farmer Food Pressure Penalty", "type": "float", "value": food_pressure_penalty},
        {"name": "Farmer Tree Knowledge Bonus", "type": "float", "value": tree_knowledge_bonus},
        {"name": "Farmer Fallback Action", "type": "int", "value": fallback_action},
    ]

def _build_gaml_parameters(
    nS,
    nA,
    nC,
    trigger_action,
    p_cut,
    difficulty,
    r_is_contextual=False,
    compliance_params=None,
):
    """
    Translates Python environment parameters into the format expected by GAMA.
    This ensures the ABM physics match the Python MDP reward distributions.
    """
    base_means = _build_base_means(nA, difficulty)
    context_scales = _build_context_scales(nC, difficulty)
    action_bonus_scales = _build_action_bonus_scales(nA)

    age_bonus_max = 0.36
    growth_rate = 3.0
    noise = 0.05 if difficulty == "easy" else 0.15

    params = [
        {"name": "Number of States", "type": "int", "value": nS},
        {"name": "Number of Actions", "type": "int", "value": nA},
        {"name": "Number of Contexts", "type": "int", "value": nC},
        {"name": "Trigger Action", "type": "int", "value": trigger_action},
        {"name": "Cut Probability", "type": "float", "value": p_cut},
        
        {"name": "Age Bonus Max", "type": "float", "value": age_bonus_max},
        {"name": "Growth Rate", "type": "float", "value": growth_rate},
        {"name": "Reward Noise", "type": "float", "value": noise},
        {"name": "Base Means", "type": "list", "value": base_means},

        {"name": "Reward Is Contextual", "type": "bool", "value": r_is_contextual},
        {"name": "Context Scales", "type": "list", "value": context_scales},
        {"name": "Action Bonus Scales", "type": "list", "value": action_bonus_scales},
    ]

    if compliance_params is None:
        compliance_params = {}

    params.extend(_build_compliance_params(**compliance_params))

    return params

def build_gama_agnostic_agrocarbon_config(
    nS=4, nA=4, nC=3, trigger_action=2,
    p_cut=0.0, difficulty="easy",
    compliance_params=None,
    gaml_experiment_path=None,
    gaml_experiment_name="gym_env",
    gama_ip_address=None,
    gama_port=None,
):
    """
    Build config for a GAMA-backed agnostic agrocarbon environment.
    """
    # Resolve GAMA connection from env vars if not provided
    if gama_ip_address is None:
        gama_ip_address = os.environ.get("GAMA_HOST", "localhost")
    if gama_port is None:
        gama_port = int(os.environ.get("GAMA_PORT", 6868))

    gaml_params = _build_gaml_parameters(nS, nA, nC, trigger_action, p_cut, difficulty, r_is_contextual=False, compliance_params=compliance_params)

    return {
        # --- MDP kwargs ---
        "nS": nS,
        "nA": nA,
        "nC": nC,
        "P": build_tree_transitions(nS, nA, trigger_action, p_cut),
        "R": build_agnostic_reward_matrix(nS, nA, nC, difficulty),
        "mu0": build_initial_state_dist(nS, nC, start_state=0),
        "nu": build_context_dist(nC),
        "skeleton": build_tree_skeleton(nS, nA, trigger_action),
        "c_is_static": True,
        "p_is_contextual": False,
        "r_is_contextual": False,
        "nameActions": build_action_names(nA),
        "seed": 123,
        
        # --- GAMA kwargs ---
        "gaml_experiment_path": gaml_experiment_path,
        "gaml_experiment_name": gaml_experiment_name,
        "gama_ip_address": gama_ip_address,
        "gama_port": gama_port,
        "gaml_experiment_parameters": gaml_params, # Injected to sync physics!
    }


def build_gama_reward_contextual_agrocarbon_config(
    nS=4, nA=4, nC=3, trigger_action=2,
    p_cut=0.0, difficulty="easy",
    compliance_params=None,
    gaml_experiment_path=None,
    gaml_experiment_name="gym_env",
    gama_ip_address=None,
    gama_port=None,
):
    """
    Build config for a GAMA-backed reward_contextual agrocarbon environment.
    """
    if gama_ip_address is None:
        gama_ip_address = os.environ.get("GAMA_HOST", "localhost")
    if gama_port is None:
        gama_port = int(os.environ.get("GAMA_PORT", 6868))

    gaml_params = _build_gaml_parameters(nS, nA, nC, trigger_action, p_cut, difficulty, r_is_contextual=True, compliance_params=compliance_params)

    return {
        # --- MDP kwargs ---
        "nS": nS,
        "nA": nA,
        "nC": nC,
        "P": build_tree_transitions(nS, nA, trigger_action, p_cut),
        "R": build_contextual_reward_matrix(nS, nA, nC, difficulty), # Contextual R!
        "mu0": build_initial_state_dist(nS, nC, start_state=0),
        "nu": build_context_dist(nC),
        "skeleton": build_tree_skeleton(nS, nA, trigger_action),
        "c_is_static": True,
        "p_is_contextual": False,
        "r_is_contextual": True, # Contextual Flag!
        "nameActions": build_action_names(nA),
        "seed": 123,
        
        # --- GAMA kwargs ---
        "gaml_experiment_path": gaml_experiment_path,
        "gaml_experiment_name": gaml_experiment_name,
        "gama_ip_address": gama_ip_address,
        "gama_port": gama_port,
        "gaml_experiment_parameters": gaml_params, 
    }