"""
Sequential experiment runner for GAMA-backed environments.

Same logic as runLargeMulticoreExperiment but without Parallel/Joblib.
Each replicate runs sequentially, reusing the same GAMA connection via env.reset().

The oracle runs on a pure-Python environment (no GAMA) built from the same
local P/R matrices, since it needs millions of steps and socket overhead
would be prohibitive.

Collects both cumulative regret and action distributions for analysis.
"""

import os
import copy
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statisticalrl_experiments.oneRun as oR
import statisticalrl_experiments.analyzeRuns as aR
import statisticalrl_experiments.plotResults as plR

from src.contextual_stat_rl.environments.ContextualMDPs_discrete.contextualMDP import ContextualDiscreteMDP
from src.contextual_stat_rl.experiments.oneRun import one_xp_run_with_actions_and_dump, dump


def build_oracle_env(gama_env):
    """
    Build a pure-Python ContextualDiscreteMDP from a ContextualGamaEnv.
    """
    return ContextualDiscreteMDP(
        nS=gama_env.nS,
        nA=gama_env.nA,
        P=gama_env.P,
        R=gama_env.R,
        mu0=gama_env.mu0,
        nC=gama_env.nC,
        nu=gama_env.nu,
        skeleton=gama_env.skeleton,
        c_is_static=gama_env.c_is_static,
        p_is_contextual=gama_env.p_is_contextual,
        r_is_contextual=gama_env.r_is_contextual,
        nameActions=getattr(gama_env, 'nameActions', []),
        seed=None,
        name=gama_env.name,
    )

def oracle_contextual_with_dump(oracle_env, oracle, timeHorizon, nbReplicates, root_folder):
    """
    Estimate the oracle cumulative mean curve on the local Python env,
    averaging over multiple resets/contexts.
    """
    curves = []

    for rep in range(nbReplicates):
        observation, info = oracle_env.reset(seed=10_000 + rep)
        oracle.reset(observation)

        cummean = 0.0
        cummeans = []

        for t in range(timeHorizon):
            state = observation
            action = oracle.play(state)

            observation, reward, done, truncated, info = oracle_env.step(action)
            oracle.update(state, action, reward, observation)

            try:
                cummean += info["mean"]
            except (TypeError, KeyError):
                cummean += reward

            cummeans.append(cummean)

            if done:
                observation, info = oracle_env.reset(seed=10_000 + rep + t + 1)
                oracle.reset(observation)

        curves.append(cummeans)

    mean_curve = np.mean(curves, axis=0)

    tag = (
        oracle_env.name + "_"
        + oracle.name() + "_"
        + str(timeHorizon) + "_"
        + str(time.time())
    )

    filename = dump([mean_curve.tolist()], "cumMeans", tag, root_folder)
    return filename


def sequentialRuns(env, learner, nbReplicates, timeHorizon, root_folder):
    """
    Run multiple replicates sequentially, collecting rewards and action counts.

    Returns
    -------
    tuple (list, float, np.ndarray)
        - cumRewardFiles: list of filenames for cumMeans dumps
        - meanElapsedTime: average time per replicate
        - avg_action_counts: mean action counts across replicates, shape (nC, nA)
    """
    cumRewardFiles = []
    all_action_counts = []
    all_time_action_counts = []
    t0 = time.time()

    for i in range(nbReplicates):
        learner_copy = copy.deepcopy(learner)
        filename, action_counts, time_action_counts = one_xp_run_with_actions_and_dump(
            env, learner_copy, timeHorizon, root_folder
        )
        cumRewardFiles.append(filename)
        all_action_counts.append(action_counts)
        all_time_action_counts.append(time_action_counts)

    elapsed = time.time() - t0
    avg_action_counts = np.mean(all_action_counts, axis=0)
    time_action_counts_sum = np.sum(all_time_action_counts, axis=0)
    time_action_freqs = time_action_counts_sum / np.maximum(
        time_action_counts_sum.sum(axis=1, keepdims=True),
        1,
    )

    return cumRewardFiles, elapsed / nbReplicates, avg_action_counts, time_action_freqs


# ==========================================
# Action Distribution Plots
# ==========================================

def plot_action_distribution(names, action_counts_per_agent, action_names,
                              title, output_path):
    """
    Grouped bar chart of action frequencies per agent (aggregated over contexts).
    """
    nAgents = len(names)
    nA = len(action_names)

    frequencies = []
    for counts in action_counts_per_agent:
        total = counts.sum()
        if total > 0:
            frequencies.append(counts.sum(axis=0) / total)
        else:
            frequencies.append(np.zeros(nA))

    x = np.arange(nAgents)
    width = 0.8 / nA
    colors = plt.cm.Set2(np.linspace(0, 1, nA))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(nA):
        offsets = x + (i - nA / 2 + 0.5) * width
        vals = [freq[i] for freq in frequencies]
        ax.bar(offsets, vals, width, label=action_names[i], color=colors[i])

    ax.set_xlabel("Agent")
    ax.set_ylabel("Action frequency")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend(title="Actions")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[PLOT] Action distribution saved to {output_path}")


def plot_action_distribution_by_context(names, action_counts_per_agent, action_names,
                                         nC, title, output_path):
    """
    Action frequencies per agent, decomposed by context (one subplot per context).
    """
    nAgents = len(names)
    nA = len(action_names)
    colors = plt.cm.Set2(np.linspace(0, 1, nA))

    fig, axes = plt.subplots(1, nC, figsize=(6 * nC, 6), sharey=True)
    if nC == 1:
        axes = [axes]

    for c in range(nC):
        ax = axes[c]

        frequencies = []
        for counts in action_counts_per_agent:
            total_c = counts[c].sum()
            if total_c > 0:
                frequencies.append(counts[c] / total_c)
            else:
                frequencies.append(np.zeros(nA))

        x = np.arange(nAgents)
        width = 0.8 / nA

        for i in range(nA):
            offsets = x + (i - nA / 2 + 0.5) * width
            vals = [freq[i] for freq in frequencies]
            ax.bar(offsets, vals, width, label=action_names[i] if c == 0 else "", color=colors[i])

        ax.set_xlabel("Agent")
        ax.set_title(f"Context {c}")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Action frequency")
    axes[0].legend(title="Actions", loc="upper left")
    fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[PLOT] Action distribution by context saved to {output_path}")

def plot_action_heatmaps_over_time(
    names,
    time_action_freqs_per_agent,
    action_names,
    title,
    output_path,
    cmap="PuBu",
):
    """
    Heatmap of action frequencies over time for each agent.

    Parameters
    ----------
    names : list[str]
        Agent names.
    time_action_freqs_per_agent : list[np.ndarray]
        One array per agent, each of shape (timeHorizon, nA).
        Entry [t, a] is the frequency of action a at timestep t.
    action_names : list[str]
        Names of the actions.
    title : str
        Figure title.
    output_path : str
        Path where the figure is saved.
    cmap : str
        Matplotlib colormap. Default is "PuBu".
    """
    n_agents = len(names)
    nA = len(action_names)

    fig, axes = plt.subplots(
        n_agents,
        1,
        figsize=(10, 2.8 * n_agents),
        sharex=True,
        constrained_layout=True,
    )

    if n_agents == 1:
        axes = [axes]

    vmax = 1.0
    vmin = 0.0

    for i, (ax, agent_name, freqs) in enumerate(
        zip(axes, names, time_action_freqs_per_agent)
    ):
        # freqs shape: (timeHorizon, nA)
        # imshow expects rows as y-axis, so transpose to actions × time
        im = ax.imshow(
            freqs.T,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(agent_name)
        ax.set_yticks(np.arange(nA))
        ax.set_yticklabels(action_names)
        ax.set_ylabel("Action")

        if i == n_agents - 1:
            ax.set_xlabel("Timestep")

    fig.suptitle(title, fontsize=14)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("Action frequency")

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[PLOT] Action heatmaps over time saved to {output_path}")


# ==========================================
# Main Experiment Runner
# ==========================================

def runSequentialGamaExperiment(
    env,
    agents,
    oracle,
    timeHorizon=1000,
    opttimeHorizon=None,
    nbReplicates=100,
    root_folder="results/",
    oracle_env=None,
):
    """
    Sequential experiment runner for GAMA environments.

    Produces:
    - Cumulative regret plots in {root_folder}/regret/
    - Action distribution plots in {root_folder}/actions/
    """
    # Create output directories
    regret_folder = os.path.join(root_folder, "regret") + os.sep
    actions_folder = os.path.join(root_folder, "actions") + os.sep
    os.makedirs(regret_folder, exist_ok=True)
    os.makedirs(actions_folder, exist_ok=True)

    envFullName = env.name
    action_names = getattr(env, 'nameActions', [f"a{i}" for i in range(env.nA)])

    learners = [x[0](**x[1]) for x in agents]

    print("*********************************************")

    dump_cumRewardsAlgos = []
    names = []
    meanelapsedtimes = []
    all_action_counts = []
    all_time_action_freqs = []

    for learner in learners:
        names.append(learner.name())
        print(f"[SEQ] Running {learner.name()} ({nbReplicates} replicates, horizon={timeHorizon})...")

        dump_cumRewards, meanelapsedtime, avg_action_counts, time_action_freqs = sequentialRuns(
            env,
            learner,
            nbReplicates,
            timeHorizon,
            root_folder=regret_folder,
        )

        dump_cumRewardsAlgos.append(dump_cumRewards)
        meanelapsedtimes.append(meanelapsedtime)
        all_action_counts.append(avg_action_counts)
        all_time_action_freqs.append(time_action_freqs)

        print(f"[SEQ] {learner.name()} done. Avg time per replicate: {meanelapsedtime:.2f}s")

    # Oracle runs on pure-Python env
    if opttimeHorizon is None:
        opttimeHorizon = min(max(1000000, timeHorizon), 10**8)

    if oracle_env is None:
        print("[SEQ] Building local oracle environment from GAMA env...")
        oracle_env = build_oracle_env(env)

    if hasattr(oracle, "env") and oracle.env is not oracle_env:
        print("[WARNING] oracle.env is not oracle_env. Make sure the oracle was built on the local oracle_env.")

    print("[SEQ] Running oracle on local env...")

    oracle_t0 = time.time()

    dump_cumRewardsopt = oracle_contextual_with_dump(
        oracle_env,
        oracle,
        timeHorizon=timeHorizon,
        nbReplicates=nbReplicates,
        root_folder=regret_folder,
    )

    oracle_elapsed = time.time() - oracle_t0
    oracle_mean_elapsed = oracle_elapsed / nbReplicates

    dump_cumRewardsAlgos.append(dump_cumRewardsopt)

    print(
        f"[SEQ] Oracle done. "
        f"Total time: {oracle_elapsed:.2f}s | "
        f"Avg time per replicate: {oracle_mean_elapsed:.4f}s"
    )

    # --- Regret analysis and plots ---
    timestamp = str(time.time())
    logfilename = regret_folder + "logfile_" + env.name + "_" + timestamp + ".txt"

    with open(logfilename, "w") as logfile:
        logfile.write("Environment " + env.name + "\n")
        logfile.write("Optimal policy is: " + str(oracle.policy) + "\n")
        logfile.write("Learners " + str([learner.name() for learner in learners]) + "\n")
        logfile.write(
            "Time horizon is " + str(timeHorizon)
            + ", nb of replicates is " + str(nbReplicates) + "\n"
        )

        for i in range(len(names)):
            logfile.write(
                str(names[i]) + " average runtime is "
                + str(meanelapsedtimes[i]) + "\n"
            )

        logfile.write(
            "Oracle total runtime is " + str(oracle_elapsed) + "\n"
        )
        logfile.write(
            "Oracle average runtime per replicate is " + str(oracle_mean_elapsed) + "\n"
        )

        mean, median, quantile1, quantile2, times = aR.computeCumulativeRegrets(
            names, dump_cumRewardsAlgos, timeHorizon, envFullName, root_folder=regret_folder
        )

        title = f"{env.name}"

        plR.plotCumulativeRegrets(
            names, envFullName, title, mean, median, quantile1, quantile2, times,
            timeHorizon, logfile=logfile, timestamp=timestamp, root_folder=regret_folder
        )

    # --- Action distribution plots ---
    plot_action_distribution(
        names, all_action_counts, action_names,
        title=f"Action Distribution — {env.name}",
        output_path=os.path.join(actions_folder, f"action_distribution_{env.name}_{timestamp}.png"),
    )

    plot_action_distribution_by_context(
        names, all_action_counts, action_names,
        nC=env.nC,
        title=f"Action Distribution by Context — {env.name}",
        output_path=os.path.join(actions_folder, f"action_distribution_by_context_{env.name}_{timestamp}.png"),
    )

    plot_action_heatmaps_over_time(
        names,
        all_time_action_freqs,
        action_names,
        title=f"Action Frequencies Over Time — {env.name}",
        output_path=os.path.join(
            actions_folder,
            f"action_heatmaps_over_time_{env.name}_{timestamp}.png",
        ),
        cmap="PuBu",
    )

    oR.clear_auxiliaryfiles(env, regret_folder)
    print(f"\n[INFO] Log-file: {logfilename}")
    print(f"[INFO] Regret plots: {regret_folder}")
    print(f"[INFO] Action plots: {actions_folder}")