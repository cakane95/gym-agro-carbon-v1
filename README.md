# GymAgroCarbon

**GymAgroCarbon** is a reinforcement learning benchmark designed to evaluate RL algorithms under combined frictions that arise in real-world decision systems: contextual heterogeneity, delayed returns, and decision-mediated execution. We formalize this setting as a **contextual Markov decision process with corrupted action execution**, where recommended actions are filtered through a structured compliance mechanism before being executed.

The motivating application is Soil Organic Carbon (SOC) management in Sahelian agroforestry. In this setting, a policymaker repeatedly recommends land-use practices across heterogeneous parcels. Parcel responses depend on local conditions such as soil type, agroforestry interventions can generate delayed ecological benefits, and farmers may not always follow the recommended actions.

GymAgroCarbon provides a family of configurable environments capturing:

1. **Contextual heterogeneity**: the value of actions depends on observed parcel contexts.
2. **Delayed ecological returns**: tree protection can initiate a multi-season maturation process that changes future rewards.
3. **Decision-mediated execution**: the learner recommends an action, but the action actually executed may be filtered by external agents.

The benchmark features a dual implementation:

- a **lightweight Python backend** for controlled tabular evaluation, oracle computation, and fast debugging;
- a **GAMA agent-based simulation backend** that models structured action corruption through cognitive farmer agents.

In the GAMA backend, Belief--Desire--Intention (BDI) farmer agents may accept or override RL recommendations based on stylized socioeconomic and cognitive factors such as household pressure and knowledge of tree benefits. This makes it possible to evaluate algorithms not only as ideal MDP controllers, but also as recommendation policies whose actions must survive a realistic implementation layer.

## Environment family

GymAgroCarbon currently provides three contextual-dependence regimes:

- `agrocarbon-agnostic`: contexts are observed but do not affect rewards or transitions.
- `agrocarbon-reward-contextual`: contexts affect rewards but not transitions.
- `agrocarbon-fully-contextual`: contexts affect both rewards and transitions through context-dependent tree persistence.

Each regime can be combined with static or dynamic context dynamics and with four difficulty scenarios: easy deterministic, easy stochastic, hard deterministic, and hard stochastic.

The corresponding GAMA-backed environments are:

- `gama-agrocarbon-agnostic`
- `gama-agrocarbon-reward-contextual`
- `gama-agrocarbon-fully-contextual`

## Docker architecture

GymAgroCarbon runs through a Docker-based Python–GAMA architecture composed of two services:

- **`gama-headless`** runs the GAMA agent-based simulation server and exposes it through a socket connection.
- **`gym-agent`** runs the Python RL environments, learners, experiment scripts, tests, and analysis utilities.

The Python backend interacts with GAMA through a Gym-like wrapper. At each step, Python sends a recommended action to GAMA, triggers one simulation step, and reads back the resulting observation, reward, termination flags, and diagnostic information.

```text
RL learner
   ↓
ContextualGamaEnv
   ↓
GAMA headless socket API
   ↓
GAMA ABM model
   ↓
Parcel dynamics + farmer compliance + rewards
   ↓
Observation / reward / info returned to Python
```
This design keeps the learner interface close to standard Gymnasium environments while allowing execution dynamics, farmer compliance, and socio-ecological mechanisms to be modeled inside GAMA.

For long experiment batches, GAMA can be restarted between scenarios using Docker Compose. The experiment runner also writes error logs when GAMA commands fail, time out, or are interrupted

## Repository organization

The repository separates the reusable Python package, the GAMA agent-based model, tests, and paper-specific experiment assets.

```text
.
├── docker-compose.yml        # Two-service setup: GAMA headless + Python RL agent
├── Dockerfile                # Python environment used by gym-agent
├── pyproject.toml            # Python package configuration
├── README.md
│
├── src/
│   └── contextual_stat_rl/
│       ├── environments/     # Contextual MDPs, GAMA wrappers, environment factories
│       ├── learners/         # ETC, IMED-RL, UCRL3, Q-learning, oracle controllers
│       └── experiments/      # Sequential runners, regret analysis, plotting utilities
│
├── gama_models/
│   └── EcoSysML/             # GAML model used by the GAMA backend
│
├── tests/                    # Integration, learner, dynamic-context, and mini-experiment tests
│
├── examples/                 # Small Python-only examples
│
└── articles/
    └── neurips26/
        ├── configs/          # YAML experiment configurations
        ├── scripts/          # Experiment launch and analysis scripts
        └── results/          # Generated outputs, usually ignored or regenerated
```

The main reusable code lives in `src/contextual_stat_rl/`. The GAMA model is located under `gama_models/EcoSysML/`, with `main.gaml` as the entry point used by the Dockerized GAMA service. Paper-specific configurations and scripts are kept under `articles/neurips26/` so that benchmark code and submission-specific experiments remain separated.

The repository currently includes both Python-only and GAMA-backed environments. The Python backend is useful for fast debugging and oracle computation, while the GAMA backend is used when compliance-aware execution and BDI farmer agents are required.

## Quick start

## Prerequisites

GymAgroCarbon is designed to run through Docker. The recommended setup requires:

- Docker
- Docker Compose
- Git

No local installation of GAMA is required when using Docker. The GAMA headless server is launched as a Docker service.

## Installation

Clone the repository and build the Docker services:

```bash
git clone <anonymous-repository-url>
cd gym-agro-carbon
docker-compose up -d --build
```

Start the two Docker services:

```bash
docker-compose up -d --build
```
This starts:

`gama-headless`: the GAMA simulation server;
`gym-agent`: the Python environment used to run tests, experiments, and analysis scripts.

To check that the containers are running:

```bash
docker-compose ps
```

### Run tests

Tests should be run in order. Each test validates one layer of the Python--GAMA pipeline.

```bash
# 1. Verify connectivity to the GAMA headless server
docker-compose exec gym-agent python tests/test_handshake.py

# 2. Validate the ContextualGamaEnv reset/step/close cycle
docker-compose exec gym-agent python tests/test_rl_cycle.py

# 3. Validate dynamic context resampling in the GAMA backend
docker-compose exec gym-agent python tests/test_dynamic_context_rl_cycle.py

# 4. Verify that fully-contextual transition kernels depend on context
docker-compose exec gym-agent python tests/test_fully_contextual_transitions.py

# 5. Verify that all RL agents can interface with the environment
docker-compose exec gym-agent python tests/test_rl_learners.py

# 6. Run a minimal experiment (agnostic, short horizon)
docker-compose exec gym-agent python tests/test_mini_experiment.py

# 7. Run a minimal experiment (reward-contextual, short horizon)
docker-compose exec gym-agent python tests/test_mini_contextual_reward_experiment.py
```

The first tests validate the GAMA connection and environment cycle. The dynamic-context and fully-contextual tests check the benchmark variants used to define the full environment family. The learner and mini-experiment tests then verify that the RL agents can run end-to-end with the environment.

A YAML-based smoke test is also provided to check the full experiment runner with a fully-contextual dynamic setting:

```bash
docker-compose exec gym-agent python articles/neurips26/scripts/run_experiment.py articles/neurips26/configs/test_full_dynamic_fully.yaml
```
This test is useful for validating the complete path from YAML configuration to GAMA execution, learner evaluation, logging, and result generation.

## Running a YAML scenario

Paper experiments are defined through YAML configuration files under:

```text
articles/neurips26/configs/
```

Each YAML file specifies the environment, horizon, number of replicates, GAMA farmer profile, and list of RL agents to evaluate. Detailed documentation of the YAML fields is provided in:

```text
articles/neurips26/configs/README.md
```

### GAMA backend

To run a scenario with the GAMA agent-based backend:

```bash
docker-compose exec gym-agent python articles/neurips26/scripts/run_experiment.py articles/neurips26/configs/full_compliance/scenario_1_easy_det.yaml
```

This uses the environment name declared in the YAML file, for example:

```yaml
experiment:
  name: "gama-agrocarbon-reward-contextual"
```

The main paper experiments use the reward-contextual, static-context setting:

```yaml
experiment:
  name: "gama-agrocarbon-reward-contextual"

environment:
  c_is_static: true
```

### Python-only backend

The same scenario can also be run with the lightweight Python backend:

```bash
docker-compose exec gym-agent python articles/neurips26/scripts/run_experiment.py articles/neurips26/configs/full_compliance/scenario_1_easy_det.yaml --python-only
```

In this mode, the GAMA backend is bypassed. The Python backend is useful for fast debugging, oracle checks, and backend-consistency experiments.

## Main paper setting

The main experiments use the reward-contextual, static-context setting:

```yaml
experiment:
  name: "gama-agrocarbon-reward-contextual"

environment:
  c_is_static: true
```

The benchmark also supports fully contextual dynamics and dynamic contexts:

```yaml
experiment:
  name: "gama-agrocarbon-fully-contextual"

environment:
  c_is_static: false
  context_p_cut_scale_gap: 0.05
  reference_context: 0
```

These variants are part of the benchmark family, but the main paper focuses on the reward-contextual static-context setting.

## Compliance analysis

GAMA-backed experiments write compliance records to the `compliance/` folder of each result directory. These records include the recommended action, the executed action, whether the farmer complied, and farmer-side variables.

To analyze compliance for a single scenario:

```bash
docker-compose exec gym-agent python articles/neurips26/scripts/run_compliance_analysis.py \
  articles/neurips26/results/low_compliance__scenario_4_hard_stoch
```

This produces:

```text
compliance/
├── per-agent recommended vs executed action heatmaps
├── global compliance heatmap over time
├── compliance rate by agent
└── compliance rate by recommended action
```

To compare compliance profiles for the same scenario:

```bash
docker-compose exec gym-agent python articles/neurips26/scripts/run_compliance_analysis.py \
  --compare \
  articles/neurips26/results/full_compliance__scenario_4_hard_stoch \
  articles/neurips26/results/med_compliance__scenario_4_hard_stoch \
  articles/neurips26/results/low_compliance__scenario_4_hard_stoch
```

This is useful for studying how farmer filtering changes across full, medium, and low compliance regimes. It must be ran after the single scenario is run.

## Backend runtime comparison

The benchmark provides both a GAMA backend and a lightweight Python backend. The GAMA backend is used for compliance-aware experiments, while the Python backend is useful for fast debugging, oracle checks, and backend-consistency experiments.

To compare backend runtimes for one scenario, first run the scenario with GAMA:

```bash
docker-compose exec gym-agent python articles/neurips26/scripts/run_experiment.py \
  articles/neurips26/configs/full_compliance/scenario_1_easy_det.yaml
```

Then run the same scenario with the Python-only backend:

```bash
docker-compose exec gym-agent python articles/neurips26/scripts/run_experiment.py \
  articles/neurips26/configs/full_compliance/scenario_1_easy_det.yaml \
  --python-only
```

The runtime comparison script expects result directories in pairs:

```text
<gama_result_dir> <python_result_dir>
```

Example for one scenario:

```bash
docker-compose exec gym-agent python articles/neurips26/scripts/compare_backend_runtime.py \
  articles/neurips26/results/full_compliance__scenario_1_easy_det \
  articles/neurips26/results_python/full_compliance__scenario_1_easy_det
```

Example for several scenarios:

```bash
docker-compose exec gym-agent python articles/neurips26/scripts/compare_backend_runtime.py \
  articles/neurips26/results/full_compliance__scenario_1_easy_det \
  articles/neurips26/results_python/full_compliance__scenario_1_easy_det \
  articles/neurips26/results/full_compliance__scenario_2_easy_stoch \
  articles/neurips26/results_python/full_compliance__scenario_2_easy_stoch
```

Arguments must always be passed as GAMA/Python result-directory pairs.

## Outputs and failure logs

Results are written to:

```text
articles/neurips26/results/<scenario_name>/
```

Each scenario folder contains:

```
regret/       # cumulative reward dumps, regret logs, regret plots
actions/      # action distributions and action-frequency heatmaps
compliance/   # recommended/executed actions and farmer-compliance records
```

If an experiment fails, the runner writes:

```text
experiment_error.txt
```

If the run is interrupted manually, it writes:

```text
experiment_interrupted.txt
```

If a replicate fails inside an experiment, the runner also writes:

```text
regret/failed_runs_<agent>.json
```

These logs contain the scenario metadata, learner name, replicate index, error type, error message, and traceback.

## Restarting GAMA between scenarios

For long experiment batches, it is recommended to restart the GAMA service between scenarios:

```bash
docker-compose restart gama-headless
```

Then rerun the desired YAML scenario:

```bash
docker-compose exec gym-agent python articles/neurips26/scripts/run_experiment.py <path-to-config.yaml>
```

The GAMA command timeout can be configured in the YAML file:

```yaml
gama:
  step_timeout: 30.0
```

For debugging failure handling, a shorter value such as 5.0 is useful. For full experiments, use a more conservative value such as `30.0` or `60.0`.

## Citation

Citation information will be added after submission.

## License

The source code is released under the MIT License. Documentation, figures, and paper-related text are released under CC BY 4.0 unless otherwise stated.

