import numpy as np
import scipy.stats as stat
from gymnasium import spaces
from statisticalrl_environments.MDPs_discrete.utils import categorical_sample
from statisticalrl_environments.MDPs_discrete.gymWrapper import DiscreteMDP

class ContextualDiscreteMDP(DiscreteMDP):
    """
    Extension of DiscreteMDP with a contextual observation.

    Attributes
    ----------
    nC : int
        Number of discrete contexts in the set C.
    nS : int
        Number of states in the common state space S.
    nA : int
        Number of actions in the common action space A.
    P : dict
        Transition probability kernels P^c. 
        If `p_is_contextual` is True, accessed via P[c][s][a]. 
        Otherwise, accessed via P[s][a].
    R : dict
        Reward distributions or mean rewards r^c. 
        If `r_is_contextual` is True, accessed via R[c][s][a]. 
        Otherwise, accessed via R[s][a].
    nu : list or np.ndarray
        Probability distribution over contexts (icd). Shape (nC,).
    mu0 : dict, list or np.ndarray
        Initial state distributions for each context c. Shape (nC, nS).
    p_is_contextual : bool
        Whether transitions depend on the context.
    r_is_contextual : bool
        Whether rewards depend on the context.
    c_is_static : bool
        If True, context is sampled once per episode. 
        If False, context is sampled at each step h.

    Methods
    -------
    __init__(...)
        Initializes the contextual MDP and defines the tuple observation space.
    reset(seed=None, options=None)
        Samples the initial context `c` and state `s`. Returns `((c, s), info)`.
    step(a)
        Executes action `a`, computes the next state and reward. 
        Resamples `c` if `c_is_static` is False. Returns `((c, s), r, done, truncated, info)`.
    """

    def __init__(
        self,
        nS,
        nA,
        P,  # Non-default arguments must come first
        R,
        mu0,
        nC,
        nu,
        skeleton=None,  # Moved down here!
        c_is_static=True,
        p_is_contextual=True,
        r_is_contextual=False,
        nameActions=None,
        seed=None,
        name="ContextualDiscreteMDP",
    ):
        if nameActions is None:
            nameActions = []

        self.nC = nC
        self.nu = nu
        self.mu0 = mu0
        
        self.c_is_static = c_is_static
        self.p_is_contextual = p_is_contextual
        self.r_is_contextual = r_is_contextual
        
        self.c = None

        if skeleton is None:
            # Default: All actions available in all states
            self.skeleton = {s: list(range(nA)) for s in range(nS)}
        else:
            self.skeleton = skeleton

        # Dummy isd
        dummy_isd = [1.0 / nS] * nS

        super().__init__(
            nS=nS,
            nA=nA,
            P=P,
            R=R,
            isd=dummy_isd,
            nameActions=nameActions,
            seed=seed,
            name=name,
        )

        # Contextual observation is strictly a tuple (context, state)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.nC),
            spaces.Discrete(self.nS),
        ))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.c = categorical_sample(self.nu, self.np_random)
        # Override parent's initial state using the context-dependent mu0
        self.s = categorical_sample(self.mu0[self.c], self.np_random)
        
        self.lastaction = None

        return (self.c, self.s), {"mean": 0.0}

    def step(self, a):
        # 1. Resolve Transitions
        if self.p_is_contextual:
            transitions = self.P[self.c][self.s][a]
        else:
            transitions = self.P[self.s][a]

        # 2. Resolve Rewards
        if self.r_is_contextual:
            rewarddis = self.R[self.c][self.s][a]
        else:
            rewarddis = self.R[self.s][a]

        # 3. Compute next state and step physics
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, next_s, done = transitions[i]

        reward = rewarddis.rvs()
        mean_reward = rewarddis.mean()

        # 4. State updates
        self.s = next_s

        # Resample context if it is dynamic
        if not self.c_is_static:
            self.c = categorical_sample(self.nu, self.np_random)

        self.lastaction = a
        self.lastreward = reward

        return (self.c, self.s), reward, done, False, {"mean": mean_reward}
    
    # -------------------------------------------------------------------------
    # Helper overrides for RL Learners (Ensuring they use the context correctly)
    # -------------------------------------------------------------------------
    
    def getTransition(self, s, a, c=None):
        target_c = c if c is not None else self.c
        transition = np.zeros(self.nS)
        
        trans_list = self.P[target_c][s][a] if self.p_is_contextual else self.P[s][a]
            
        for t in trans_list:
            transition[t[1]] = t[0]
        return transition

    def getMeanReward(self, s, a, c=None):
        target_c = c if c is not None else self.c
        rewarddis = self.R[target_c][s][a] if self.r_is_contextual else self.R[s][a]
        return rewarddis.mean()