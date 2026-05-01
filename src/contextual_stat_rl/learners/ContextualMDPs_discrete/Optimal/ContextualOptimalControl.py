import numpy as np
from statisticalrl_learners.MDPs_discrete.Optimal.OptimalControl import Opti_controller
from statisticalrl_environments.MDPs_discrete.utils import categorical_sample
from statisticalrl_learners.MDPs_discrete.utils import allmax

def build_opti(name, env, nS, nA):
    """
    Build the appropriate oracle based on the environment's contextual flags.
    
    If r_is_contextual or p_is_contextual, uses ContextualOpti_controller
    which solves a separate VI per context. Otherwise, uses GlobalOpti_controller.
    """
    r_ctx = getattr(env, 'r_is_contextual', False)
    p_ctx = getattr(env, 'p_is_contextual', False)
    nC = getattr(env, 'nC', 1)

    if r_ctx or p_ctx:
        return ContextualOpti_controller(env, nS, nA, nC)
    else:
        return GlobalOpti_controller(env, nS, nA)


class GlobalOpti_controller(Opti_controller):
    """
    Global optimal controller for contextual environments.

    Computes a single optimal policy over states, ignoring contexts.
    Appropriate when rewards and transitions are context-independent.
    """

    def __init__(self, env, nS, nA, epsilon=0.001, max_iter=100):
        super().__init__(env, nS, nA, epsilon=epsilon, max_iter=max_iter)
        self.observation = None
        self.c = None
        self.s = None

    def parse_observation(self, observation):
        c, s = observation[0], observation[1]
        return c, s

    def reset(self, observation):
        self.observation = observation
        self.c, self.s = self.parse_observation(observation)

    def play(self, observation):
        self.observation = observation
        self.c, self.s = self.parse_observation(observation)
        a = categorical_sample([self.policy[self.s, a] for a in range(self.nA)], np.random)
        return a

    def update(self, observation, action, reward, next_observation):
        self.observation = next_observation
        self.c, self.s = self.parse_observation(next_observation)


class ContextualOpti_controller:
    """
    Contextual optimal controller that solves one VI per context.

    Computes policy[c][s, a] by solving a separate MDP for each context c,
    using context-specific rewards (and optionally context-specific transitions).

    This is the true oracle for r_is_contextual and/or p_is_contextual environments.
    """

    def __init__(self, env, nS, nA, nC, epsilon=1e-7, max_iter=100000, gamma=0.999):
        self.env = env
        self.nS = nS
        self.nA = nA
        self.nC = nC
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.gamma = gamma

        self.observation = None
        self.c = None
        self.s = None

        # Per-context arrays
        self.transitions = np.zeros((nC, nS, nA, nS))
        self.meanrewards = np.zeros((nC, nS, nA))
        self.policy = np.zeros((nC, nS, nA))
        self.u = np.zeros((nC, nS))

        # Build transition and reward matrices per context
        for c in range(nC):
            for s in range(nS):
                for a in range(nA):
                    self.transitions[c, s, a] = env.getTransition(s, a, c)
                    self.meanrewards[c, s, a] = env.getMeanReward(s, a, c)
                    self.policy[c, s, a] = 1.0 / nA

            # Solve VI for this context
            self._vi_for_context(c)

    def _vi_for_context(self, c):
        """Run value iteration for a single context."""
        u0 = np.zeros(self.nS)
        u1 = np.zeros(self.nS)

        for itera in range(self.max_iter):
            for s in range(self.nS):
                temp = np.zeros(self.nA)
                for a in range(self.nA):
                    temp[a] = (
                        self.meanrewards[c, s, a]
                        + self.gamma * self.transitions[c, s, a] @ u0
                    )
                best_val, choice = allmax(temp)
                u1[s] = best_val
                self.policy[c, s] = [
                    1.0 / len(choice) if x in choice else 0
                    for x in range(self.nA)
                ]

            diff = np.abs(u1 - u0)
            if (max(diff) - min(diff)) < self.epsilon:
                self.u[c] = u1 - min(u1)
                break

            u0 = u1 - min(u1)
            u1 = np.zeros(self.nS)

    def name(self):
        return "Contextual_Optimal_controller"

    def parse_observation(self, observation):
        c, s = observation[0], observation[1]
        return c, s

    def reset(self, observation):
        self.observation = observation
        self.c, self.s = self.parse_observation(observation)

    def play(self, observation):
        self.c, self.s = self.parse_observation(observation)
        a = categorical_sample(
            [self.policy[self.c, self.s, a] for a in range(self.nA)],
            np.random,
        )
        return a

    def update(self, observation, action, reward, next_observation):
        self.observation = next_observation
        self.c, self.s = self.parse_observation(next_observation)