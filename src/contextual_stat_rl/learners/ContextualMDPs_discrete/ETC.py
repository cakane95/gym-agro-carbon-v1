import copy
import numpy as np
from .ContextualAgentInterface import ContextualAgent

class ETC(ContextualAgent):
    """
    Base class for Explore-Then-Commit (ETC) learners in the cMDP setting.

    This class implements the common ETC workflow: an exploration phase used to
    collect empirical reward and transition estimates, followed by a commit phase
    based on a policy inferred from the explored data.

    Subclasses must implement `explore()`, `commit()`, `stop_exploration()`,
    and `build_committed_policy()`.

    Attributes
    ----------
    skeleton : dict
        A dictionary mapping each state `s` to a 1D array of admissible actions.
    exploration_phase : bool
        Flag indicating whether the agent is currently exploring or has committed.
    committed_policy : dict or np.ndarray
        The fixed policy used during the commit phase (built once exploration stops).
    state_action_pulls : np.ndarray
        Visit counts for (context, state, action) or (state, action) pairs.
    state_visits : np.ndarray
        Visit counts for (context, state) or (state) pairs.
    rewards : np.ndarray
        Empirical mean rewards. Initialized to 0.5.
    transitions : np.ndarray
        Empirical transition probabilities. Initialized to a uniform distribution.
    transition_pulls : np.ndarray
        Visit counts specifically tracking transitions.
    """
    def __init__(self, nS, nA, nC, skeleton=None, learning_scope="global", name="ETC_Base"):
        super().__init__(nS, nA, nC, learning_scope=learning_scope, name=name)

        self.actions = np.arange(self.nA, dtype=int)

        if skeleton is None:
            self.skeleton = {s: np.arange(self.nA, dtype=int) for s in range(self.nS)}
        else:
            self.skeleton = self._validate_and_copy_skeleton(skeleton)

        self.initial_skeleton = copy.deepcopy(self.skeleton)

        # Rewards and visit counts
        if self.uses_context_for_rewards():
            self.state_action_pulls = np.zeros((self.nC, self.nS, self.nA), dtype=int)
            self.state_visits = np.zeros((self.nC, self.nS), dtype=int)
            self.rewards = np.zeros((self.nC, self.nS, self.nA)) + 0.5
        else:
            self.state_action_pulls = np.zeros((self.nS, self.nA), dtype=int)
            self.state_visits = np.zeros(self.nS, dtype=int)
            self.rewards = np.zeros((self.nS, self.nA)) + 0.5

        # Transitions
        if self.uses_context_for_transitions():
            self.transitions = np.ones((self.nC, self.nS, self.nA, self.nS)) / self.nS
            self.transition_pulls = np.zeros((self.nC, self.nS, self.nA), dtype=int)
        else:
            self.transitions = np.ones((self.nS, self.nA, self.nS)) / self.nS
            self.transition_pulls = np.zeros((self.nS, self.nA), dtype=int)

        self.exploration_phase = True
        self.committed_policy = None

    def _validate_and_copy_skeleton(self, skeleton):
        validated = {}

        if set(skeleton.keys()) != set(range(self.nS)):
            raise ValueError(f"skeleton must define admissible actions for all states 0..{self.nS - 1}")

        for s in range(self.nS):
            actions = np.array(skeleton[s], dtype=int)

            if actions.ndim != 1:
                raise ValueError(f"skeleton[{s}] must be a 1D array-like of actions")

            if len(actions) == 0:
                raise ValueError(f"skeleton[{s}] must contain at least one action")

            if np.any(actions < 0) or np.any(actions >= self.nA):
                raise ValueError(f"skeleton[{s}] contains invalid action indices")

            validated[s] = np.unique(actions)

        return validated

    def reset(self, observation):
        super().reset(observation)
        self.exploration_phase = True
        self.committed_policy = None
        self.skeleton = copy.deepcopy(self.initial_skeleton)

        self.state_action_pulls.fill(0)
        self.state_visits.fill(0)
        self.transition_pulls.fill(0)
        self.rewards.fill(0.5)
        self.transitions.fill(1.0 / self.nS)

    def get_available_actions(self, observation=None):
        s = self.get_state(observation)
        return self.skeleton[s]

    def is_action_available(self, action, observation=None):
        return action in self.get_available_actions(observation)

    def play(self, observation):
        if self.exploration_phase:
            if self.stop_exploration():
                self.exploration_phase = False
                self.build_committed_policy()
                return self.commit(observation)
            else:
                return self.explore(observation)
        else:
            return self.commit(observation)

    def update(self, observation, action, reward, next_observation):
        if self.exploration_phase:
            self.update_model(observation, action, reward, next_observation)

        self.observation = next_observation
        self.c, self.s = self.parse_observation(next_observation)

    def update_model(self, observation, action, reward, next_observation):
        if not self.is_action_available(action, observation):
            raise ValueError(
                f"Action {action} is not admissible in state {self.get_state(observation)}"
            )

        r_key = self.get_reward_key(observation)
        t_key = self.get_transition_key(observation)
        next_s = self.get_state(next_observation)

        idx_r = (*r_key, action) if isinstance(r_key, tuple) else (r_key, action)
        idx_t = (*t_key, action) if isinstance(t_key, tuple) else (t_key, action)
        idx_v = r_key

        na_r = self.state_action_pulls[idx_r]
        self.state_action_pulls[idx_r] += 1
        self.state_visits[idx_v] += 1
        self.rewards[idx_r] = ((na_r * self.rewards[idx_r]) + reward) / (na_r + 1)

        na_t = self.transition_pulls[idx_t]
        self.transition_pulls[idx_t] += 1

        dirac_next_s = np.zeros(self.nS)
        dirac_next_s[next_s] = 1.0
        self.transitions[idx_t] = ((na_t * self.transitions[idx_t]) + dirac_next_s) / (na_t + 1)

    def explore(self, observation):
        raise NotImplementedError("explore() must be implemented by subclasses.")

    def commit(self, observation):
        raise NotImplementedError("commit() must be implemented by subclasses.")

    def stop_exploration(self):
        raise NotImplementedError("stop_exploration() must be implemented by subclasses.")

    def build_committed_policy(self):
        raise NotImplementedError("build_committed_policy() must be implemented by subclasses.")

class GlobalETC(ETC):
    """
    Explore-Then-Commit agent (Global scope) that explores for exactly exploration_limit timesteps.
    
    At each exploration step, it chooses the least played valid action in the 
    current state. After exploration_limit steps, it computes the optimal policy via Value 
    Iteration on the empirical MDP and commits to it.

    Attributes
    ----------
    exploration_limit : int
        The total number of exploratory steps before committing (set to 4).
    gamma : float
        Discount factor used in Value Iteration.
    epsilon : float
        Convergence threshold for Value Iteration.
    max_iter : int
        Maximum number of iterations allowed for Value Iteration.

    Methods
    -------
    explore(observation)
        Selects the valid action with the minimum number of pulls in the current state.
    stop_exploration()
        Returns True when the total number of pulls reaches the exploration limit.
    build_committed_policy()
        Runs Value Iteration to compute and store the optimal deterministic policy.
    commit(observation)
        Returns the action dictated by the committed policy for the current state.
    """
    def __init__(
        self, nS, nA, nC, skeleton=None,
        gamma=0.99, epsilon=1e-3, max_iter=1000,
        exploration_limit=4, name=None,
    ):
        if name is None:
            name = f"GlobalETC{exploration_limit}"
        super().__init__(nS, nA, nC, skeleton=skeleton, learning_scope="global", name=name)
        self.exploration_limit = exploration_limit
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iter = max_iter

    def explore(self, observation):
        """
        Chooses the valid action that has been pulled the least number of times 
        in the current state.
        """
        s = self.get_state(observation)
        valid_actions = self.get_available_actions(observation)
        
        valid_pulls = self.state_action_pulls[s, valid_actions]
        min_pulls = np.min(valid_pulls)
        candidate_actions = valid_actions[valid_pulls == min_pulls]
        return np.random.choice(candidate_actions)

    def stop_exploration(self):
        """
        Stops strictly when the total number of exploratory actions taken reaches the limit.
        """
        return np.sum(self.state_action_pulls) >= self.exploration_limit

    def build_committed_policy(self):
        """
        Runs Value Iteration on the empirical MDP to find the optimal policy.
        """
        V = np.zeros(self.nS)
        self.committed_policy = np.zeros(self.nS, dtype=int)
        
        for _ in range(self.max_iter):
            Q = np.full((self.nS, self.nA), -np.inf)
            
            for s in range(self.nS):
                for a in self.skeleton[s]:
                    r = self.rewards[s, a]
                    p = self.transitions[s, a]
                    Q[s, a] = r + self.gamma * np.dot(p, V)
            
            new_V = np.max(Q, axis=1)
            
            if np.max(np.abs(V - new_V)) < self.epsilon:
                V = new_V
                break
                
            V = new_V
            
        for s in range(self.nS):
            Q_s = np.full(self.nA, -np.inf)
            for a in self.skeleton[s]:
                Q_s[a] = self.rewards[s, a] + self.gamma * np.dot(self.transitions[s, a], V)
            
            max_q = np.max(Q_s[Q_s > -np.inf])
            best_actions = np.where(Q_s == max_q)[0]
            self.committed_policy[s] = np.random.choice(best_actions)

    def commit(self, observation):
        """
        Executes the pre-computed deterministic policy.
        """
        s = self.get_state(observation)
        return self.committed_policy[s]