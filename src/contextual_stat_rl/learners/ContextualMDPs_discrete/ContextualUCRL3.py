import warnings
import numpy as np

from .ContextualAgentInterface import (
    ContextualAgent,
)

from statisticalrl_learners.MDPs_discrete.UCRL3 import UCRL3


class GlobalUCRL3(ContextualAgent):
    """
    Contextual adapter for the standard UCRL3 algorithm.

    This version is global/context-agnostic:
    - observations are contextual pairs (c, s);
    - UCRL3 only receives the state s;
    - rewards are optionally normalized before being passed to UCRL3.

    This makes UCRL3 compatible with ContextualGamaEnv while keeping the
    original UCRL3 implementation unchanged.
    """

    def __init__(
        self,
        nS,
        nA,
        nC,
        delta=0.05,
        K=-1,
        max_reward=2.5,
        name="GlobalUCRL3",
    ):
        super().__init__(
            nS=nS,
            nA=nA,
            nC=nC,
            learning_scope="global",
            name=name,
        )

        self.delta = delta
        self.K = K
        self.max_reward = max_reward

        self.core = UCRL3(
            nS=nS,
            nA=nA,
            delta=delta,
            K=K,
        )

    def name(self):
        return "GlobalUCRL3"

    def reset(self, observation):
        super().reset(observation)

        s = self.get_state(observation)
        
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in scalar divide",
                category=RuntimeWarning,
            )
            self.core.reset(s)

    def play(self, observation):
        self.observation = observation
        self.c, self.s = self.parse_observation(observation)

        s = self.get_state(observation)
        return self.core.play(s)

    def update(self, observation, action, reward, next_observation):
        self.observation = next_observation
        self.c, self.s = self.parse_observation(next_observation)

        s = self.get_state(observation)
        next_s = self.get_state(next_observation)

        # UCRL3 assumes rewards roughly in [0, 1].
        # We normalize the observed reward to keep the original confidence bounds valid.
        reward_normalized = reward / self.max_reward
        reward_normalized = float(np.clip(reward_normalized, 0.0, 1.0))

        self.core.update(
            s,
            action,
            reward_normalized,
            next_s,
        )