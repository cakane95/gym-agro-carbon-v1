"""
Compliance analysis for GymAgroCarbon experiments.

Reads compliance CSV files generated during GAMA-backed runs and produces:
- per-agent recommended vs executed action heatmaps;
- global compliance heatmap over time;
- compliance rate by agent;
- compliance rate by recommended action;
- optional comparison across compliance profiles.

Usage
-----
Single scenario:
    docker-compose exec gym-agent python articles/neurips26/scripts/run_compliance_analysis.py \
        articles/neurips26/results/low_compliance__scenario_4_hard_stoch

Compare profiles:
    docker-compose exec gym-agent python articles/neurips26/scripts/run_compliance_analysis.py \
        --compare \
        articles/neurips26/results/full_compliance__scenario_4_hard_stoch \
        articles/neurips26/results/medium_compliance__scenario_4_hard_stoch \
        articles/neurips26/results/low_compliance__scenario_4_hard_stoch
"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ACTION_NAMES = {
    0: "fallow",
    1: "fert_fallow",
    2: "tree",
    3: "baseline",
}


# ============================================================
# Loading utilities
# ============================================================

def load_compliance_csv(result_dir):
    """
    Load all compliance CSV files from a result directory.

    Expected directory:
        result_dir/compliance/*.csv
    """
    compliance_dir = os.path.join(result_dir, "compliance")
    files = glob.glob(os.path.join(compliance_dir, "*.csv"))

    if not files:
        raise FileNotFoundError(f"No compliance CSV found in {compliance_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = os.path.basename(f)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    # Normalize booleans if needed
    if out["complied"].dtype != bool:
        out["complied"] = out["complied"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )

    return out


def get_scenario_name(result_dir):
    """Return the last folder name as scenario/profile identifier."""
    return os.path.basename(os.path.normpath(result_dir))


def ensure_output_dir(result_dir):
    """Create analysis output directory."""
    output_dir = os.path.join(result_dir, "compliance_analysis")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ============================================================
# Core summaries
# ============================================================

def summarize_compliance(df):
    """
    Print basic compliance summaries.
    """
    print("\n=== Compliance rate by agent ===")
    print(df.groupby("agent")["complied"].mean())

    print("\n=== Compliance rate by recommended action ===")
    print(
        df.groupby(["agent", "action_recommended"])["complied"]
        .mean()
        .unstack(fill_value=np.nan)
        .rename(columns=ACTION_NAMES)
    )

    print("\n=== Recommended vs executed actions ===")
    table = pd.crosstab(
        df["action_recommended"],
        df["action_executed"],
        normalize="index",
    )
    table = table.rename(index=ACTION_NAMES, columns=ACTION_NAMES)
    print(table)


# ============================================================
# Plot 1: recommended vs executed heatmaps per agent
# ============================================================

def _action_frequency_matrix(df_agent, action_col, n_actions=4):
    """
    Build matrix of shape (n_actions, timeHorizon).

    Entry [a, t] = frequency of action a at timestep t.
    """
    timesteps = sorted(df_agent["timestep"].unique())
    matrix = np.zeros((n_actions, len(timesteps)))

    for j, t in enumerate(timesteps):
        df_t = df_agent[df_agent["timestep"] == t]
        counts = df_t[action_col].value_counts(normalize=True)

        for a in range(n_actions):
            matrix[a, j] = counts.get(a, 0.0)

    return matrix, timesteps


def plot_recommended_vs_executed_heatmaps(
    df,
    output_dir,
    action_names=None,
    cmap="PuBu",
):
    """
    For each agent, plot two heatmaps:
    - top: recommended action frequencies over time
    - bottom: executed action frequencies over time
    """
    if action_names is None:
        action_names = [ACTION_NAMES[i] for i in range(4)]

    agents = sorted(df["agent"].unique())

    for agent in agents:
        df_agent = df[df["agent"] == agent]

        rec_matrix, timesteps = _action_frequency_matrix(
            df_agent,
            "action_recommended",
            n_actions=len(action_names),
        )
        exe_matrix, _ = _action_frequency_matrix(
            df_agent,
            "action_executed",
            n_actions=len(action_names),
        )

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10, 5.5),
            sharex=True,
            constrained_layout=True,
        )

        im0 = axes[0].imshow(
            rec_matrix,
            aspect="auto",
            origin="upper",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        axes[0].set_title(f"{agent} — Recommended actions")
        axes[0].set_yticks(np.arange(len(action_names)))
        axes[0].set_yticklabels(action_names)
        axes[0].set_ylabel("Action")

        im1 = axes[1].imshow(
            exe_matrix,
            aspect="auto",
            origin="upper",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        axes[1].set_title(f"{agent} — Executed actions")
        axes[1].set_yticks(np.arange(len(action_names)))
        axes[1].set_yticklabels(action_names)
        axes[1].set_ylabel("Action")
        axes[1].set_xlabel("Timestep")

        axes[1].set_xticks(np.arange(len(timesteps)))
        axes[1].set_xticklabels(timesteps)

        fig.colorbar(im1, ax=axes, shrink=0.85, label="Action frequency")

        safe_agent = agent.replace(" ", "_").replace("/", "_")
        output_path = os.path.join(
            output_dir,
            f"recommended_vs_executed_heatmap_{safe_agent}.png",
        )

        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"[PLOT] Saved {output_path}")


# ============================================================
# Plot 2: global compliance heatmap
# ============================================================

def plot_global_compliance_heatmap(
    df,
    output_dir,
    title="Compliance rate over time",
    cmap="RdBu",
):
    """
    Heatmap with:
    - rows = agents
    - columns = timesteps
    - color = compliance rate
    """
    pivot = (
        df.groupby(["agent", "timestep"])["complied"]
        .mean()
        .unstack(fill_value=np.nan)
    )

    agents = list(pivot.index)
    timesteps = list(pivot.columns)

    fig, ax = plt.subplots(figsize=(11, max(3, 0.6 * len(agents))))

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="upper",
        cmap=cmap,
        vmin=0,
        vmax=1,
    )

    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Agent")

    ax.set_xticks(np.arange(len(timesteps)))
    ax.set_xticklabels(timesteps)

    ax.set_yticks(np.arange(len(agents)))
    ax.set_yticklabels(agents)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Compliance rate")

    output_path = os.path.join(output_dir, "global_compliance_heatmap.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[PLOT] Saved {output_path}")


# ============================================================
# Plot 3: compliance rate by action
# ============================================================

def plot_compliance_by_action(df, output_dir):
    """
    Barplot of compliance rate by recommended action and agent.
    """
    grouped = (
        df.groupby(["agent", "action_recommended"])["complied"]
        .mean()
        .reset_index()
    )

    agents = sorted(df["agent"].unique())
    actions = sorted(df["action_recommended"].unique())

    x = np.arange(len(agents))
    width = 0.8 / len(actions)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, a in enumerate(actions):
        vals = []
        for agent in agents:
            sub = grouped[
                (grouped["agent"] == agent)
                & (grouped["action_recommended"] == a)
            ]
            vals.append(float(sub["complied"].iloc[0]) if len(sub) else np.nan)

        offsets = x + (i - len(actions) / 2 + 0.5) * width
        ax.bar(offsets, vals, width, label=ACTION_NAMES.get(a, str(a)))

    ax.set_xlabel("Agent")
    ax.set_ylabel("Compliance rate")
    ax.set_title("Compliance rate by recommended action")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.legend(title="Recommended action")

    output_path = os.path.join(output_dir, "compliance_rate_by_action.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[PLOT] Saved {output_path}")


# ============================================================
# Optional comparison across profiles
# ============================================================

def compare_profiles(result_dirs):
    """
    Compare several result directories, typically:
    full_compliance, medium_compliance, low_compliance
    for the same scenario.
    """
    rows = []

    for result_dir in result_dirs:
        label = get_scenario_name(result_dir)
        df = load_compliance_csv(result_dir)

        summary = (
            df.groupby("agent")["complied"]
            .mean()
            .reset_index()
            .rename(columns={"complied": "compliance_rate"})
        )
        summary["profile"] = label
        rows.append(summary)

    out = pd.concat(rows, ignore_index=True)
    return out


def plot_profile_comparison(summary_df, output_dir):
    """
    Compare average compliance rate by agent across profiles.
    """
    profiles = list(summary_df["profile"].unique())
    agents = list(summary_df["agent"].unique())

    x = np.arange(len(agents))
    width = 0.8 / len(profiles)

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, profile in enumerate(profiles):
        vals = []
        for agent in agents:
            sub = summary_df[
                (summary_df["profile"] == profile)
                & (summary_df["agent"] == agent)
            ]
            vals.append(float(sub["compliance_rate"].iloc[0]) if len(sub) else np.nan)

        offsets = x + (i - len(profiles) / 2 + 0.5) * width
        ax.bar(offsets, vals, width, label=profile)

    ax.set_xlabel("Agent")
    ax.set_ylabel("Average compliance rate")
    ax.set_title("Compliance comparison across profiles")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.legend(title="Profile")

    output_path = os.path.join(output_dir, "profile_compliance_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[PLOT] Saved {output_path}")


# ============================================================
# Main
# ============================================================

def run_single_analysis(result_dir):
    """
    Analyze one result directory.
    """
    scenario_name = get_scenario_name(result_dir)
    output_dir = ensure_output_dir(result_dir)

    print("=" * 60)
    print(f"Compliance analysis: {scenario_name}")
    print(f"Result dir: {result_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    df = load_compliance_csv(result_dir)
    summarize_compliance(df)

    plot_recommended_vs_executed_heatmaps(
        df,
        output_dir=output_dir,
        cmap="PuBu",
    )

    plot_global_compliance_heatmap(
        df,
        output_dir=output_dir,
        title=f"Compliance Rate Over Time — {scenario_name}",
        cmap="RdBu",
    )

    plot_compliance_by_action(df, output_dir=output_dir)

    print("\n[DONE] Single compliance analysis completed.")


def run_comparison(result_dirs):
    """
    Compare compliance profiles across result directories.
    """
    common_parent = os.path.commonpath(result_dirs)
    output_dir = os.path.join(common_parent, "compliance_profile_comparison")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Compliance profile comparison")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    summary_df = compare_profiles(result_dirs)

    summary_path = os.path.join(output_dir, "compliance_profile_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[CSV] Saved {summary_path}")

    plot_profile_comparison(summary_df, output_dir)

    print("\n[DONE] Compliance comparison completed.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_dirs",
        nargs="+",
        help="One or more result directories containing compliance CSVs.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare several compliance profiles instead of analyzing one directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.compare:
        if len(args.result_dirs) < 2:
            raise ValueError("--compare requires at least two result directories.")
        run_comparison(args.result_dirs)
    else:
        if len(args.result_dirs) != 1:
            raise ValueError("Single analysis expects exactly one result directory.")
        run_single_analysis(args.result_dirs[0])


if __name__ == "__main__":
    main()