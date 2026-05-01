"""
Custom run functions for GAMA-backed experiments.

Extends the standard statisticalrl_experiments.oneRun with action tracking
for generating action distribution plots.
"""

import os
import time
import pickle
import numpy as np


def dump(values, filename, tag, root_folder):
    os.makedirs(root_folder, exist_ok=True)
    filenameM = os.path.join(root_folder, f"{filename}_{tag}")
    with open(filenameM, "wb") as file:
        pickle.dump(values, file)
    return filenameM


def one_xp_run_with_actions_and_dump(env, learner, timeHorizon, root_folder):
    """
    Run one episode, collecting cumulative mean rewards, action counts,
    action counts over time, and compliance records.

    Tracks:
    - action_counts[c, a]: how many times action a was recommended in context c
    - time_action_counts[t, a]: how many times action a was recommended at timestep t
    - compliance_records: one row per timestep for compliance analysis

    Returns
    -------
    tuple
        - filename: path to the dumped cumMeans file
        - action_counts: array of shape (nC, nA)
        - time_action_counts: array of shape (timeHorizon, nA)
        - compliance_records: list[dict]
    """
    observation, info = env.reset()
    learner.reset(observation)

    nC = env.nC if hasattr(env, "nC") else env.unwrapped.nC
    nA = env.nA if hasattr(env, "nA") else env.unwrapped.nA

    cumreward = 0.0
    cumrewards = []
    cummean = 0.0
    cummeans = []

    action_counts = np.zeros((nC, nA), dtype=int)
    time_action_counts = np.zeros((timeHorizon, nA), dtype=int)
    compliance_records = []

    print(f"[Info] New initialization of {learner.name()} for environment {env.name}")

    for t in range(timeHorizon):
        state = observation
        action = learner.play(state)

        if isinstance(state, tuple) and len(state) == 2:
            c = int(state[0])
            s = int(state[1])
        else:
            c = 0
            s = None

        # Here action is the recommendation sent by the RL learner.
        action_counts[c, action] += 1
        time_action_counts[t, action] += 1

        observation, reward, done, truncated, info = env.step(action)
        learner.update(state, action, reward, observation)

        cumreward += reward
        try:
            cummean += info["mean"]
        except (TypeError, KeyError):
            cummean += reward

        cumrewards.append(cumreward)
        cummeans.append(cummean)

        action_recommended = info.get("action_recommended", action)
        action_executed = info.get("action_executed", action)
        complied = info.get("complied", action_recommended == action_executed)

        compliance_records.append({
            "timestep": t,
            "context": c,
            "state": s,
            "action_recommended": int(action_recommended),
            "action_executed": int(action_executed),
            "complied": bool(complied),
            "reward": float(reward),
            "mean": float(info.get("mean", reward)),
            "was_cut": bool(info.get("was_cut", False)),
            "compliance_probability": info.get("compliance_probability", None),
            "household_size": info.get("household_size", None),
            "tree_knowledge": info.get("tree_knowledge", None),
        })

        if done:
            print(f"Episode finished after {t + 1} timesteps")
            observation, info = env.reset()
            learner.reset(observation)

    tag = env.name + "_" + learner.name() + "_" + str(timeHorizon) + "_" + str(time.time())
    filename = dump(cummeans, "cumMeans", tag, root_folder)

    return filename, action_counts, time_action_counts, compliance_records