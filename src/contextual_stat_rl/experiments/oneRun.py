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
    and action counts over time.

    Tracks:
    - action_counts[c, a]: how many times action a was played in context c
    - time_action_counts[t, a]: how many times action a was played at timestep t

    Returns
    -------
    tuple (str, np.ndarray, np.ndarray)
        - filename: path to the dumped cumMeans file
        - action_counts: array of shape (nC, nA)
        - time_action_counts: array of shape (timeHorizon, nA)
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

    print(f"[Info] New initialization of {learner.name()} for environment {env.name}")

    for t in range(timeHorizon):
        state = observation
        action = learner.play(state)

        # Track the action in its context
        if isinstance(state, tuple) and len(state) == 2:
            c = state[0]
        else:
            c = 0

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

        if done:
            print(f"Episode finished after {t + 1} timesteps")
            observation, info = env.reset()
            learner.reset(observation)

    tag = env.name + "_" + learner.name() + "_" + str(timeHorizon) + "_" + str(time.time())
    filename = dump(cummeans, "cumMeans", tag, root_folder)

    return filename, action_counts, time_action_counts