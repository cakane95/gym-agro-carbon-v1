import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


class BatchContextualMDP(gym.Env):
    """Generic environment for Batched Contextual MDPs with vectorized execution.

    This class inherits from `EnvP` and extends it to handle multiple 
    contextual episodes executed in parallel within discrete batches. It combines 
    the contextual structure 
    with the limited adaptivity constraint. 

    The environment is "vectorized": it manages B episodes simultaneously, 
    where B is the size of the current batch. The state and observation 
    spaces are arrays of size B, allowing for full parallelization during 
    each batch.

    Attributes:
        nC (int): Number of discrete contexts in the set C.
        nS (int): Number of states in the common state space S.
        nA (int): Number of actions in the common action space A.
        H (int): Finite horizon of a single episode (steps per episode).
        K (int): Total budget of episodes (global budget).
        M (int): Maximum number of batches allowed.
        
        P (np.ndarray): Transition probability kernels P^c. 
            If `p_is_contextual` is True, shape is (nC, nS, nA, nS). 
            Otherwise, shape is (nS, nA, nS).
        R (np.ndarray): Reward distributions or mean rewards r^c. 
            If `r_is_contextual` is True, shape is (nC, nS, nA). 
            Otherwise, shape is (nS, nA).
        
        nu (np.ndarray): Probability distribution over contexts (icd). 
            Shape (nC,).
        mu0 (np.ndarray): Initial state distributions for each context c. 
            Shape (nC, nS).
        
        p_is_contextual (bool): Whether transitions depend on the context.
        r_is_contextual (bool): Whether rewards depend on the context.
        c_is_static (bool): If True, context is sampled once per episode. 
            If False, context is sampled at each step h.
        tau_is_static (bool): If True, the batch grid tau is fixed at 
            initialization. If False, it is adaptive.
            
        current_k (int): Global index of the total episodes completed.
        current_m (int): Index of the current batch j(k).
        current_h (int): Current step within the episodes of the batch.
        
        current_batch_c (np.ndarray): Vector of contexts for the current batch. 
            Shape (B,).
        current_batch_s (np.ndarray): Vector of states for the current batch. 
            Shape (B,).
            
        history (List[List[dict]]): Global history H_m containing data from 
            all completed batches. 
            Structure: Batch -> Episode -> {context, steps}.
        batch_buffer (List[dict]): Temporary storage for the B trajectories 
            of the ongoing batch, ensuring H_m-measurability.

    Properties:
        tau (np.ndarray): The batch grid boundaries (tau_0, ..., tau_M).
        batch_limit (int): Episode index tau_m marking the end of the current batch.
        is_at_batch_boundary (bool): True if the current batch is completed (k = tau_m).
        next_batch_size (int): Size B of the upcoming batch (tau_m - tau_{m-1}).
    """
    def __init__(
        self,
        nC, nS, nA, H, K, M,
        P, R, nu, mu0,
        p_is_contextual=True,
        r_is_contextual=True,
        c_is_static=True,
        tau_is_static=True,
        tau=None,
        nameActions=None,
        seed=None,
        name="BatchContextualMDP"
    ):
        """Initializes the Batched Contextual MDP."""
        # Sanity checks
        assert M <= K, f"M={M} must be <= K={K}"
        assert nu.shape == (nC,) and np.isclose(nu.sum(), 1.0)
        assert mu0.shape == (nC, nS)
        assert np.allclose(mu0.sum(axis=1), 1.0)
        if p_is_contextual:
            assert P.shape == (nC, nS, nA, nS)
        else:
            assert P.shape == (nS, nA, nS)
        if r_is_contextual:
            assert R.shape == (nC, nS, nA)
        else:
            assert R.shape == (nS, nA)

        # Identity and dimensions
        self.name = name
        self.nC, self.nS, self.nA = nC, nS, nA
        self.H, self.K, self.M = H, K, M
        
        # Model
        self.P = P  # Transition kernels P^c 
        self.R = R  # Reward distributions r^c
        self.nu = nu  # Context distribution nu
        self.mu0 = mu0  # Initial state distributions mu_0^c
        
        # Protocol Flags
        self.p_is_contextual = p_is_contextual
        self.r_is_contextual = r_is_contextual
        self.c_is_static = c_is_static
        self.tau_is_static = tau_is_static
        
        # Batch Grid Initialization
        if self.tau_is_static:
            if tau is not None:
                self._tau = np.array(tau)
            else:
                # Perchet-style geometric grid: tau_m ~ K^(m/M)
                ratios = np.array([self.K ** (m / self.M) for m in range(self.M + 1)])
                self._tau = np.round(ratios).astype(int)
                self._tau[0], self._tau[-1] = 0, self.K
        else:
            # For adaptive grids, only tau_0 = 0 is known
            self._tau = np.zeros(self.M + 1, dtype=int)
            self._tau[0] = 0

        # Internal Counters & History
        self.current_k = 0
        self.current_m = 0
        self.current_h = None
        self.history = []      # H_m (public)
        self.batch_buffer = [] # Current batch trajectories (private)
        
        # Vectorized State Tracking
        self.current_batch_c = None  # (B,)
        self.current_batch_s = None  # (B,)

        # Gymnasium Spaces
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.nC),
            spaces.Discrete(self.nS),
        ))

        # Action Naming
        if nameActions is None or len(nameActions) != nA:
            import string
            self.nameActions = list(string.ascii_uppercase)[0:min(nA, 26)]
        else:
            self.nameActions = nameActions

        # Seeding
        self._np_random = None
        self.seed(seed)
        #self.reset()

    def seed(self, seed=None):
        """Standard seeding logic."""
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def tau(self):
        return self._tau

    @property
    def batch_limit(self):
        """Returns tau_m: the end boundary of the current batch."""
        return self._tau[self.current_m + 1]

    @property
    def next_batch_size(self):
        """Calculates B = tau_{m+1} - tau_m."""
        return self.batch_limit - self._tau[self.current_m]

    @property
    def is_at_batch_boundary(self):
        """Checks if the global episode counter has reached the batch limit."""
        return self.current_k >= self.batch_limit

    def reset(self, seed=None, options=None):
        """Resets the environment for a new batch of episodes."""
        super().reset(seed=seed)
        
        B = self.next_batch_size
        self.current_batch_c = self.np_random.choice(self.nC, size=B, p=self.nu)
        
        self.current_batch_s = np.array([
            self.np_random.choice(self.nS, p=self.mu0[c]) 
            for c in self.current_batch_c
        ])
        
        # Initialize History Buffers for the batch
        self.current_h = 0
        self.batch_buffer = [{"context": c, "steps": []} for c in self.current_batch_c]
        
        info = {"nextbatchsize": B, "mean": 0.0}
        return (self.current_batch_c, self.current_batch_s), info

    
    def step(self, actions):
        """Executes a vector of actions for the entire batch.
        
        Args:
            actions (np.ndarray): An array of B actions, one for each episode in the batch.
            
        Returns:
            Tuple: (observations, rewards, done, truncated, info)
        """
        B = self.next_batch_size
        assert len(actions) == B, f"Expected {B} actions, got {len(actions)}"

        batch_rewards = np.zeros(B)
        batch_means = np.zeros(B)
        next_states = np.zeros(B, dtype=int)
        
        for i in range(B):
            c_i = self.current_batch_c[i]
            s_i = self.current_batch_s[i]
            a_i = actions[i]

            # Get reward distribution R^c
            if self.r_is_contextual:
                reward_dist = self.R[c_i, s_i, a_i]
            else:
                reward_dist = self.R[s_i, a_i]
            
            batch_rewards[i] = reward_dist.rvs(random_state=self.np_random)
            batch_means[i] = reward_dist.mean()

            # Get transition probabilities P^c
            if self.p_is_contextual:
                p_dist = self.P[c_i, s_i, a_i]
            else:
                p_dist = self.P[s_i, a_i]
            
            next_states[i] = self.np_random.choice(self.nS, p=p_dist)

        # Store transitions in the private batch_buffer
        for i in range(B):
            self.batch_buffer[i]["steps"].append({
                "state": self.current_batch_s[i],
                "action": actions[i],
                "reward": batch_rewards[i],
                "next_state": next_states[i]
            })

        # Update internal state
        self.current_h += 1
        self.current_batch_s = next_states
        
        # If not static, re-sample contexts
        if not self.c_is_static:
            self.current_batch_c = self.np_random.choice(self.nC, size=B, p=self.nu)

        # 4. Handle Episode and Batch Boundaries
        done = self.current_h >= self.H
        if done:
            self.current_k += B # Move global counter by batch size
            
            # Check if we reached tau_m (Batch Boundary)
            if self.is_at_batch_boundary:
                self.history.append(list(self.batch_buffer))
                self.current_m += 1
                # The next reset() will handle clearing the buffer and resizing for B_m+1
        
        
        info = {
            "mean": np.sum(batch_means),
            "nextbatchsize": self.next_batch_size if not done else B,
            "current_m": self.current_m,
            "current_k": self.current_k
        }

        return (self.current_batch_c, self.current_batch_s), batch_rewards, done, False, info