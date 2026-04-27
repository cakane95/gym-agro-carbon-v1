import numpy as np
from statisticalrl_learners.MDPs_discrete.Optimal.OptimalControl import Opti_controller
from statisticalrl_environments.MDPs_discrete.utils import categorical_sample

def build_opti(name, env, nS, nA):
    return GlobalOpti_controller(env, nS, nA)

class GlobalOpti_controller(Opti_controller):
    """
    Global optimal controller for contextual environments.

    The controller receives contextual observations of the form (c, s), but
    computes and applies a policy depending only on the MDP state s. It acts
    as an adapter, allowing a standard state-based optimal controller to 
    interface seamlessly with a ContextualDiscreteMDP.

    Attributes
    ----------
    observation : tuple
        The most recent observation tuple (c, s).
    c : int or None
        The most recently observed context.
    s : int or None
        The most recently observed state.

    Methods
    -------
    parse_observation(observation)
        Extracts the context `c` and state `s` from the observation tuple.
    reset(observation)
        Resets the controller's internal observation tracking.
    play(observation)
        Samples and returns an action from the optimal policy for the current state `s`.
    update(observation, action, reward, next_observation)
        Updates the controller's tracked observation for the next step.
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