"""
Sequential experiment runner for GAMA-backed environments.

Same logic as runLargeMulticoreExperiment but without Parallel/Joblib.
Each replicate runs sequentially, reusing the same GAMA connection via env.reset().

The oracle runs on a pure-Python environment (no GAMA) built from the same
local P/R matrices, since it needs millions of steps and socket overhead
would be prohibitive.

Compatible with the same agents, oracle, and analysis/plotting pipeline.
"""

import os
import copy
import time
import numpy as np

import statisticalrl_experiments.oneRun as oR
import statisticalrl_experiments.analyzeRuns as aR
import statisticalrl_experiments.plotResults as plR

from src.contextual_stat_rl.environments.ContextualMDPs_discrete.contextualMDP import ContextualDiscreteMDP


def build_oracle_env(gama_env):
    """
    Build a pure-Python ContextualDiscreteMDP from a ContextualGamaEnv.

    Extracts P, R, mu0, nu, skeleton, and all flags from the GAMA env
    and creates a lightweight local environment for the oracle to run
    millions of steps without socket overhead.
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
        nameActions=gama_env.nameActions if hasattr(gama_env, 'nameActions') else [],
        seed=None,
        name=gama_env.name,
    )


def sequentialRuns(env, learner, nbReplicates, timeHorizon, oneRunFunction, root_folder):
    """
    Run multiple replicates sequentially, reusing the same env with reset().

    Unlike multicoreRuns which creates new env instances via gymnasium.make(),
    this reuses the same ContextualGamaEnv and resets between replicates.
    """
    cumRewards = []
    t0 = time.time()

    for i in range(nbReplicates):
        learner_copy = copy.deepcopy(learner)
        result = oneRunFunction(env, learner_copy, timeHorizon, root_folder)
        cumRewards.append(result)

    elapsed = time.time() - t0
    return cumRewards, elapsed / nbReplicates


def runSequentialGamaExperiment(
    env,
    agents,
    oracle,
    timeHorizon=1000,
    opttimeHorizon=10000,
    nbReplicates=100,
    root_folder="results/",
    oracle_env=None,
):
    """
    Sequential version of runLargeMulticoreExperiment for GAMA environments.

    Same interface, same outputs (logfile, plots, regret analysis).
    Differences:
    - Replicates run sequentially instead of in parallel.
    - Oracle runs on a pure-Python env (no GAMA overhead).

    Parameters
    ----------
    env : ContextualGamaEnv
        The GAMA-backed environment for agent runs.
    agents : list of (class, kwargs) tuples
        Agent definitions.
    oracle : Opti_controller
        Pre-built oracle. Should be built on oracle_env, not on env.
    oracle_env : ContextualDiscreteMDP, optional
        Pure-Python env for oracle runs. If None, built from env's local P/R.
    """
    os.makedirs(root_folder, exist_ok=True)

    envFullName = env.name

    learners = [x[0](**x[1]) for x in agents]

    print("*********************************************")

    dump_cumRewardsAlgos = []
    names = []
    meanelapsedtimes = []

    for learner in learners:
        names.append(learner.name())
        print(f"[SEQ] Running {learner.name()} ({nbReplicates} replicates, horizon={timeHorizon})...")

        dump_cumRewards, meanelapsedtime = sequentialRuns(
            env, learner, nbReplicates, timeHorizon,
            oR.oneXpNoRenderWithDump, root_folder=root_folder,
        )

        dump_cumRewardsAlgos.append(dump_cumRewards)
        meanelapsedtimes.append(meanelapsedtime)

        print(f"[SEQ] {learner.name()} done. Avg time per replicate: {meanelapsedtime:.2f}s")

    # Oracle runs on pure-Python env (no GAMA socket overhead)
    if opttimeHorizon is None:
        opttimeHorizon = min(max(10000, timeHorizon), 10**8)

    if oracle_env is None:
        print(f"[SEQ] Building local oracle environment from GAMA env...")
        oracle_env = build_oracle_env(env)
    
    if hasattr(oracle, "env") and oracle.env is not oracle_env:
        print("[WARNING] oracle.env is not oracle_env. Make sure the oracle was built on the local oracle_env.")

    print(f"[SEQ] Running oracle on local env...")
    dump_cumRewardsopt = oR.oneRunOptWithDump(oracle_env, oracle, opttimeHorizon, root_folder=root_folder)
    dump_cumRewardsAlgos.append(dump_cumRewardsopt)

    # Report statistics and compute regret
    timestamp = str(time.time())
    logfilename = root_folder + "logfile_" + env.name + "_" + timestamp + ".txt"

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

        mean, median, quantile1, quantile2, times = aR.computeCumulativeRegrets(
            names, dump_cumRewardsAlgos, timeHorizon, envFullName, root_folder=root_folder
        )

        title = f"{env.name}"

        plR.plotCumulativeRegrets(
            names, envFullName, title, mean, median, quantile1, quantile2, times,
            timeHorizon, logfile=logfile, timestamp=timestamp, root_folder=root_folder
        )

    oR.clear_auxiliaryfiles(env, root_folder)
    print(f"\n[INFO] A log-file has been generated in {logfilename}")