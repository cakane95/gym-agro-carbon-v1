import numpy as np

from .ContextualAgentInterface import ContextualAgent


class GlobalQLearning(ContextualAgent):
    """
    Global tabular Q-learning agent for contextual MDP observations.

    This learner receives observations of the form (c, s), but learns a
    context-agnostic Q-table Q[s, a]. It is therefore a global baseline:
    contexts are observed but ignored for value learning.

    Parameters
    ----------
    nS : int
        Number of states.
    nA : int
        Number of actions.
    nC : int
        Number of contexts.
    gamma : float, default=0.99
        Discount factor.
    epsilon : float, default=0.1
        Initial epsilon for epsilon-greedy exploration.
    epsilon_min : float, default=0.01
        Minimum exploration rate.
    epsilon_decay : float, default=0.995
        Multiplicative decay applied after each update.
    alpha : float or None, default=None
        Learning rate. If None, uses visit-dependent alpha = 1 / N(s, a).
    optimistic_init : float, default=0.0
        Initial value for Q-table.
    name : str, default="GlobalQLearning"
        Agent name.
    """

    def __init__(
        self,
        nS,
        nA,
        nC,
        gamma=0.99,
        epsilon=0.1,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        alpha=None,
        optimistic_init=0.0,
        name="GlobalQLearning",
    ):
        super().__init__(
            nS=nS,
            nA=nA,
            nC=nC,
            learning_scope="global",
            name=name,
        )

        self.gamma = gamma
        self.epsilon0 = epsilon
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.optimistic_init = optimistic_init

        self.Q = np.full((self.nS, self.nA), optimistic_init, dtype=float)
        self.N = np.zeros((self.nS, self.nA), dtype=int)
        self.t = 0

    def name(self):
        return "GlobalQLearning"

    def reset(self, observation):
        """
        Reset the learner for a new episode.

        The Q-table is kept across episodes/replicates only if the same learner
        instance is reused. In your experiment runner, each replicate deep-copies
        the learner and calls reset(), so this resets the table for that run.
        """
        super().reset(observation)

        self.Q = np.full((self.nS, self.nA), self.optimistic_init, dtype=float)
        self.N = np.zeros((self.nS, self.nA), dtype=int)
        self.epsilon = self.epsilon0
        self.t = 0

    def play(self, observation):
        """
        Select an action using epsilon-greedy exploration.

        Parameters
        ----------
        observation : tuple
            Contextual observation (c, s).

        Returns
        -------
        int
            Selected action.
        """
        self.observation = observation
        self.c, self.s = self.parse_observation(observation)

        s = self.get_state(observation)

        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.nA))

        max_q = np.max(self.Q[s])
        candidates = np.flatnonzero(np.isclose(self.Q[s], max_q))
        return int(np.random.choice(candidates))

    def update(self, observation, action, reward, next_observation):
        """
        Apply one Q-learning update.

        Parameters
        ----------
        observation : tuple
            Previous observation (c, s).
        action : int
            Action recommended by the learner.
        reward : float
            Reward returned by the environment. In GAMA compliance settings,
            this is the realized reward after farmer execution.
        next_observation : tuple
            Next observation (c', s').
        """
        s = self.get_state(observation)
        next_s = self.get_state(next_observation)

        self.N[s, action] += 1

        if self.alpha is None:
            alpha_t = 1.0 / self.N[s, action]
        else:
            alpha_t = self.alpha

        td_target = reward + self.gamma * np.max(self.Q[next_s])
        td_error = td_target - self.Q[s, action]

        self.Q[s, action] += alpha_t * td_error

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

        self.observation = next_observation
        self.c, self.s = self.parse_observation(next_observation)
        self.t += 1