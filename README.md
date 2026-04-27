# GYM AGRO CARBON

This project provides a framework to instantiate contextual Markov Decision Processes and connect them to agent-based models built in GAMA. The long-term goal is to build a reinforcement-learning recommender system that interacts with BDI farmer agents, who may choose to follow or ignore the recommended actions.

## GymAgroCarbon Environments

GymAgroCarbon provides 12 contextual MDP environments for evaluating exploration strategies in agroforestry SOC management. Each environment models a Regenerative Natural Assistance (RNA) decision problem where an agent selects land-use practices (fallow, fertilized fallow, tree protection, baseline) on a parcel with heterogeneous soil contexts.

### Learning Scopes

**Agnostic** — Rewards and transitions are identical across all contexts. The context is observed but carries no information. Algorithms that pool data across contexts are expected to perform best.

**Reward-Contextual** — Rewards depend on the soil context (via per-context multipliers on base means). Transitions remain context-independent. Algorithms must learn different reward structures per context to achieve optimal regret.

**Fully-Contextual** — Both rewards and transitions depend on the context (e.g., different tree destruction probabilities per soil type). *(Planned — not yet implemented.)*

### Difficulty Scenarios

| Scenario | Difficulty | p_cut | Description |
|----------|-----------|-------|-------------|
| 1 | Easy | 0.0 | Large reward gaps (0.2–1.2), low noise (σ=0.05), deterministic transitions. Easiest to learn. |
| 2 | Easy | 0.2 | Large reward gaps, low noise, but 20% chance of tree destruction each season. Investment in RNA is risky. |
| 3 | Hard | 0.0 | Tight reward gaps (0.4–0.7), high noise (σ=0.2), deterministic transitions. Requires extensive exploration to distinguish actions. |
| 4 | Hard | 0.2 | Tight gaps, high noise, and stochastic tree destruction. The most challenging configuration. |

### Full Environment Grid

| | Easy / Det (S1) | Easy / Stoch (S2) | Hard / Det (S3) | Hard / Stoch (S4) |
|---|---|---|---|---|
| **Agnostic** | `agnostic-easy-det` | `agnostic-easy-stoch` | `agnostic-hard-det` | `agnostic-hard-stoch` |
| **Reward-Contextual** | `reward-ctx-easy-det` | `reward-ctx-easy-stoch` | `reward-ctx-hard-det` | `reward-ctx-hard-stoch` |
| **Fully-Contextual** | `full-ctx-easy-det` | `full-ctx-easy-stoch` | `full-ctx-hard-det` | `full-ctx-hard-stoch` |

## Dockerized Python–GAMA Headless Architecture

The project runs through a Docker-based setup composed of two main services. The first service runs **GAMA headless** and exposes the GAMA server through a socket connection. The second service runs the **Python RL environment**, learners, experiment scripts, and tests.

Python communicates with GAMA through the headless API: it sends actions, triggers simulation steps, reads observations and rewards, and controls experiment resets.

At a high level, the architecture is:

```text
Python RL agents
      ↓
ContextualGamaEnv
      ↓
GAMA Headless API / socket
      ↓
GAMA ABM model
      ↓
Parcel dynamics, rewards, transitions
      ↓
State / Reward / Info returned to Python
```

This allows the learning algorithms to keep a standard Gym-like interface while delegating the environmental dynamics to the GAMA agent-based model.

## Repository Organization

The repository is organized to keep the reusable Python package, GAMA models, tests, examples, and paper-specific experiments clearly separated. The `src/` directory contains the Python package, including contextual MDP environments, GAMA-backed environments, RL learners, factories, and experiment runners. The `gama_models/` directory contains the GAML model files executed by GAMA headless. The `tests/` directory contains handshake, environment-cycle, learner, and mini-experiment tests used to validate the Python–GAMA pipeline. The `examples/` directory contains simple Python-only examples, while `articles/` contains experiment configurations and scripts specific to paper submissions such as NeurIPS 2026.

```text
./
  docker-compose.yml
  Dockerfile
  poetry.lock
  poetry.toml
  pyproject.toml
  README.md
  .gitignore

  articles/
    neurips26/
      configs/
        scenario_1_easy_det.yaml
      scripts/
        run_experiment.py
        run_all.py # Not implemented yet

  examples/
    run_basic_agrocarbon.py

  gama_models/
    EcoSysML/
      includes/
      models/
        EcoSysMLBehavior.gaml
        EcoSysMLStructure.gaml
        main.gaml

  src/
    contextual_stat_rl/
      __init__.py

      experiments/
        __init__.py
        sequential_experiment.py

      environments/
        __init__.py
        register.py
        gama_register.py

        ContextualMDPs_discrete/
          __init__.py
          contextualMDP.py
          contextual_gama_env.py

          factories/
            agrocarbon_factory.py
            gama_agrocarbon_factory.py

        BatchContextualMDPs_discrete/
          # Not implemented yet

      learners/
        __init__.py

        ContextualMDPs_discrete/
          ContextualAgentInterface.py
          ContextualIMED_RL.py
          ETC.py

          Optimal/
            ContextualOptimalControl.py

  tests/
    test_handshake.py
    test_rl_cycle.py
    test_rl_learners.py
    test_mini_experiment.py
    test_mini_contextual_reward_experiment.py
```

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Git

### Installation

```bash
git clone https://github.com/cakane95/gym-agro-carbon-v1.git
cd gym-agro-carbon-v1
```

### Launch

```bash
# Build and start containers (GAMA headless + Python agent)
docker-compose up -d

# Wait for GAMA headless to initialize (~15 seconds)
sleep 15
```

### Run Tests

Tests should be run in order — each validates a layer of the pipeline.

```bash
# 1. Verify connectivity to GAMA headless server
docker-compose exec gym-agent python tests/test_handshake.py

# 2. Validate the ContextualGamaEnv reset/step/close cycle
docker-compose exec gym-agent python tests/test_rl_cycle.py

# 3. Verify that all RL agents can interface with the environment
docker-compose exec gym-agent python tests/test_rl_learners.py

# 4. Run a minimal experiment (agnostic, 20 steps, 5 replicas)
docker-compose exec gym-agent python tests/test_mini_experiment.py

# 5. Run a minimal experiment (reward-contextual, 20 steps, 5 replicas)
docker-compose exec gym-agent python tests/test_mini_contextual_reward_experiment.py
```

### Shutdown

```bash
docker-compose down
```


