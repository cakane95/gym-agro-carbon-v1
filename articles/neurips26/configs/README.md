# YAML Configuration Guide

This folder contains YAML configuration files used to run GymAgroCarbon experiments for the paper. Each file defines one scenario: the environment, horizon, number of replicates, GAMA compliance profile, and RL agents to evaluate.

A scenario can be launched with:

```bash
docker-compose exec gym-agent python articles/neurips26/scripts/run_experiment.py articles/neurips26/configs/<path-to-config.yaml>
```

The same YAML can be run with the lightweight Python backend using:

```bash
docker-compose exec gym-agent python articles/neurips26/scripts/run_experiment.py articles/neurips26/configs/<path-to-config.yaml> --python-only
```

## Main sections

Each YAML file has four main sections:

```yaml
experiment:
  name: "gama-agrocarbon-reward-contextual"
  timeHorizon: 20
  nbReplicates: 100

environment:
  nS: 8
  nA: 4
  nC: 3
  trigger_action: 2
  p_cut: 0.0
  difficulty: "easy"
  c_is_static: true

gama:
  gaml_experiment_name: "gym_env"
  step_timeout: 30.0
  farmer:
    household_size: 5
    tree_knowledge: 0.6
    base_compliance: 0.75
    food_pressure_penalty: 0.2
    tree_knowledge_bonus: 0.1
    fallback_action: 3

agents:
  - class: "GlobalETC"
    params:
      exploration_limit: 10
      gamma: 0.99
      epsilon: 1.0e-6
      max_iter: 3000
```

## `experiment`

```yaml
experiment:
  name: "gama-agrocarbon-reward-contextual"
  timeHorizon: 20
  nbReplicates: 100
```

- `name`: environment name to instantiate.
- `timeHorizon`: number of decision steps per replicate.
- `nbReplicates`: number of independent replicates.

Available Python-only environments:

```text
agrocarbon-agnostic
agrocarbon-reward-contextual
agrocarbon-fully-contextual
```

Available GAMA-backed environments:

```text
gama-agrocarbon-agnostic
gama-agrocarbon-reward-contextual
gama-agrocarbon-fully-contextual
```

The main paper experiments use:

```yaml
name: "gama-agrocarbon-reward-contextual"
```

## `environment`

```yaml
environment:
  nS: 8
  nA: 4
  nC: 3
  trigger_action: 2
  p_cut: 0.0
  difficulty: "easy"
  c_is_static: true
```

- `nS`: number of tree-age states. In the paper, `nS: 8`, so maturity corresponds to state `7`.
- `nA`: number of actions.
- `nC`: number of contexts.
- `trigger_action`: action that initiates tree protection / ANR.
- `p_cut`: baseline probability of tree destruction.
- `difficulty`: either `"easy"` or `"hard"`.
- `c_is_static`: whether the context remains fixed during an episode.

The four main scenarios are:

```text
scenario_1_easy_det    difficulty=easy, p_cut=0.0
scenario_2_easy_stoch  difficulty=easy, p_cut=0.15
scenario_3_hard_det    difficulty=hard, p_cut=0.0
scenario_4_hard_stoch  difficulty=hard, p_cut=0.15
```

For fully-contextual environments, transitions can also depend on context through context-dependent cutting risk:

```yaml
environment:
  c_is_static: false
  context_p_cut_scale_gap: 0.05
  reference_context: 0
```

- `context_p_cut_scale_gap`: multiplicative gap between context-specific cutting-risk scales.
- `reference_context`: context index whose cutting-risk scale is `1.0`.

For example, with `nC: 3`, `reference_context: 0`, and `context_p_cut_scale_gap: 0.05`, the context cutting-risk scales are:

```text
[1.00, 1.05, 1.10]
```

The main paper experiments use `c_is_static: true` and the reward-contextual setting.

## `gama`

```yaml
gama:
  gaml_experiment_name: "gym_env"
  step_timeout: 30.0
  farmer:
    household_size: 5
    tree_knowledge: 0.6
    base_compliance: 0.75
    food_pressure_penalty: 0.2
    tree_knowledge_bonus: 0.1
    fallback_action: 3
```

- `gaml_experiment_name`: GAMA experiment name in `main.gaml`.
- `step_timeout`: timeout in seconds for GAMA commands.
- `farmer`: compliance profile used by BDI farmer agents.

The compliance mechanism filters recommended actions before execution. If the farmer does not comply, the recommendation is replaced by `fallback_action`, which is conventional cropping in the paper experiments.

Typical compliance regimes are:

```text
full compliance:
  household_size: 1
  tree_knowledge: 1.0
  base_compliance: 1.0
  food_pressure_penalty: 0.0
  tree_knowledge_bonus: 0.0
  fallback_action: 3

medium compliance:
  household_size: 5
  tree_knowledge: 0.6
  base_compliance: 0.75
  food_pressure_penalty: 0.2
  tree_knowledge_bonus: 0.1
  fallback_action: 3

low compliance:
  household_size: 8
  tree_knowledge: 0.2
  base_compliance: 0.55
  food_pressure_penalty: 0.35
  tree_knowledge_bonus: 0.0
  fallback_action: 3
```

For debugging GAMA failures, use a shorter timeout such as:

```yaml
step_timeout: 5.0
```

For full experiments, use a more conservative value such as:

```yaml
step_timeout: 30.0
```

or:

```yaml
step_timeout: 60.0
```

## `agents`

The `agents` section lists the RL algorithms to evaluate.

Example:

```yaml
agents:
  - class: "GlobalETC"
    params:
      exploration_limit: 10
      gamma: 0.99
      epsilon: 1.0e-6
      max_iter: 3000

  - class: "GlobalUCRL3"
    params:
      delta: 0.05
      K: -1
      max_reward: 2.0

  - class: "GlobalQLearning"
    params:
      gamma: 0.99
      epsilon: 0.3
      epsilon_min: 0.02
      epsilon_decay: 0.98
      alpha: 0.1
      optimistic_init: 0.0

  - class: "GlobalIMEDRL"
    params:
      max_iter: 3000
      epsilon: 1.0e-3
      max_reward: 2.0

  - class: "SemiLocalIMEDRL"
    params:
      max_iter: 3000
      epsilon: 1.0e-3
      max_reward: 2.0
```

Implemented agents include:

```text
GlobalETC
GlobalUCRL3
GlobalQLearning
GlobalIMEDRL
SemiLocalIMEDRL
```

Algorithms that require an explicit reward range use:

```yaml
max_reward: 2.0
```

This is the conservative reward bound used in the paper experiments.

## Outputs

Results are written to:

```text
articles/neurips26/results/<scenario_name>/
```

Each scenario folder contains:

```text
regret/       cumulative reward dumps, regret logs, and regret plots
actions/      action distributions and action-frequency heatmaps
compliance/   recommended/executed actions and compliance records
```

If a run fails, the runner writes:

```text
experiment_error.txt
```

If a run is manually interrupted, it writes:

```text
experiment_interrupted.txt
```

If a replicate fails inside an experiment, it writes:

```text
regret/failed_runs_<agent>.json
```