import copy
import numpy as np
from scipy.optimize import minimize_scalar

from .ContextualAgentInterface import ContextualAgent


def randamax(v, t=None, i=None):
    """Return index of maximum value, breaking ties randomly or by secondary array t.

    Parameters
    ----------
    v : array-like
        Primary values.
    t : array-like, optional
        Secondary values for tie-breaking (lowest wins).
    i : array-like, optional
        Subset of indices to consider.
    """
    if i is None:
        idxs = np.where(v == np.amax(v))[0]
        if t is None:
            idx = np.random.choice(idxs)
        else:
            assert len(v) == len(t), f"Lengths should match: len(v)={len(v)} - len(t)={len(t)}"
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = idxs[t_idxs]
    else:
        idxs = np.where(v[i] == np.amax(v[i]))[0]
        if t is None:
            idx = i[np.random.choice(idxs)]
        else:
            assert len(v) == len(t), f"Lengths should match: len(v)={len(v)} - len(t)={len(t)}"
            t = t[i]
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = i[idxs[t_idxs]]
    return idx


def randamin(v, t=None, i=None):
    """Return index of minimum value, breaking ties randomly or by secondary array t.

    Parameters
    ----------
    v : array-like
        Primary values.
    t : array-like, optional
        Secondary values for tie-breaking (lowest wins).
    i : array-like, optional
        Subset of indices to consider.
    """
    if i is None:
        idxs = np.where(v == np.amin(v))[0]
        if t is None:
            idx = np.random.choice(idxs)
        else:
            assert len(v) == len(t), f"Lengths should match: len(v)={len(v)} - len(t)={len(t)}"
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = idxs[t_idxs]
    else:
        idxs = np.where(v[i] == np.amin(v[i]))[0]
        if t is None:
            idx = i[np.random.choice(idxs)]
        else:
            assert len(v) == len(t), f"Lengths should match: len(v)={len(v)} - len(t)={len(t)}"
            t = t[i]
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = i[idxs[t_idxs]]
    return idx


class ContextualIMEDRL(ContextualAgent):
    """Base class for contextual extensions of IMED-RL.

    Implements the common IMED-RL logic in the cMDP setting:
    forced exploration until all admissible actions have been tried
    in the relevant scope, dynamic skeleton pruning, Value Iteration,
    and IMED index computation.

    Subclasses must define how statistics are stored and accessed
    depending on the learning scope (global or semi-local).

    Parameters
    ----------
    nbr_states : int
        Number of states in the MDP.
    nbr_actions : int
        Number of actions.
    nbr_contexts : int
        Number of discrete contexts.
    skeleton : dict, optional
        Mapping state -> array of admissible action indices.
        Defaults to all actions in every state.
    learning_scope : str
        Either "global" or "semi-local".
    name : str
        Agent identifier.
    max_iter : int
        Maximum iterations for value iteration.
    epsilon : float
        Convergence threshold for value iteration.
    max_reward : float
        Upper bound on single-step rewards.
    """

    def __init__(
        self,
        nbr_states,
        nbr_actions,
        nbr_contexts,
        skeleton=None,
        learning_scope="global",
        name="ContextualIMED-RL",
        max_iter=3000,
        epsilon=1e-3,
        max_reward=1,
    ):
        ContextualAgent.__init__(
            self,
            nbr_states,
            nbr_actions,
            nbr_contexts,
            learning_scope=learning_scope,
            name=name,
        )

        self.nS = nbr_states
        self.nA = nbr_actions
        self.nC = nbr_contexts

        self.dirac = np.eye(self.nS, dtype=int)
        self.actions = np.arange(self.nA, dtype=int)

        self.max_iteration = max_iter
        self.epsilon = epsilon
        self.max_reward = max_reward

        if skeleton is None:
            self.skeleton = {s: np.arange(self.nA, dtype=int) for s in range(self.nS)}
        else:
            self.skeleton = self._validate_and_copy_skeleton(skeleton)

        self.initial_skeleton = copy.deepcopy(self.skeleton)
        self.index = np.zeros(self.nA)
        self.prune_counts = np.zeros((self.nS, self.nA), dtype=int)

        self._init_statistics()

    def _validate_and_copy_skeleton(self, skeleton):
        """Validate and return a deep copy of the skeleton dictionary."""
        validated = {}

        if set(skeleton.keys()) != set(range(self.nS)):
            raise ValueError(
                f"skeleton must define admissible actions for all states 0..{self.nS - 1}"
            )

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
        """Reset the agent for a new episode."""
        super().reset(observation)
        self.skeleton = copy.deepcopy(self.initial_skeleton)
        self.prune_counts = np.zeros((self.nS, self.nA), dtype=int)
        self.index = np.zeros(self.nA)
        self._reset_statistics(observation)

    def get_available_actions(self, observation=None):
        """Return the current skeleton for the state in observation."""
        s = self.get_state(observation)
        return self.skeleton[s]

    def value_iteration(self, observation):
        """Run value iteration on current empirical MDP estimates."""
        ctr = 0
        stop = False

        phi = np.copy(self.get_phi(observation))
        phip = np.copy(phi)

        while not stop:
            ctr += 1
            for state in range(self.nS):
                u = -np.inf
                for action in self.skeleton[state]:
                    psa = self.get_transition_from_state_action(observation, state, action)
                    rsa = self.get_reward_from_state_action(observation, state, action)
                    u = max(u, rsa + psa @ phi)
                phip[state] = u

            phip = phip - np.min(phip)
            delta = np.max(np.abs(phi - phip))
            phi = np.copy(phip)
            stop = (delta < self.epsilon) or (ctr >= self.max_iteration)

        self.set_phi(observation, np.copy(phi))

    def _update_skeleton(self, observation):
        """Prune the skeleton using global pull counts.

        Actions are pruned from the initial admissible set.
        If pruning would empty the skeleton, the most-pulled
        admissible action is retained as a fallback.
        """
        s = self.get_state(observation)
        max_na = np.max(self.prune_counts[s])

        if max_na > 0:
            threshold = np.log(max_na) ** 2
            admissible = self.initial_skeleton[s]
            mask = self.prune_counts[s, admissible] >= threshold
            pruned = admissible[mask]

            if len(pruned) == 0:
                best = admissible[np.argmax(self.prune_counts[s, admissible])]
                pruned = np.array([best], dtype=int)

            self.skeleton[s] = pruned

    def _update_all_selected(self, observation):
        """Check whether all actions in the current skeleton have been pulled."""
        if not self.get_all_selected(observation):
            valid_actions = self.get_available_actions(observation)
            pulls = np.array(
                [self.get_pull_count(observation, a) for a in valid_actions], dtype=int
            )
            self.set_all_selected(observation, np.all(pulls > 0))

    def update(self, observation, action, reward, next_observation):
        """Full update step: statistics, pruning, exploration flag, state transition."""
        self.update_statistics(observation, action, reward, next_observation)

        s = self.get_state(observation)
        self.prune_counts[s, action] += 1

        self._update_skeleton(observation)
        self._update_all_selected(observation)

        self.observation = next_observation
        self.c, self.s = self.parse_observation(next_observation)

    def multinomial_imed(self, observation):
        """Compute the IMED index for each action in the current skeleton."""
        valid_actions = self.get_available_actions(observation)
        phi = self.get_phi(observation)

        self.index[:] = np.inf

        q = np.full(self.nA, -np.inf)
        for a in valid_actions:
            q[a] = (
                self.get_reward_estimate(observation, a)
                + self.get_transition_estimate(observation, a) @ phi
            )

        mu = np.max(q[valid_actions])
        upper_bound = self.max_reward + np.max(phi)
        u = upper_bound / (upper_bound - mu) - 1e-2

        for a in valid_actions:
            n = self.get_pull_count(observation, a)

            if q[a] >= mu:
                self.index[a] = np.log(max(n, 1))
            else:
                r_d = self.get_reward_distribution(observation, a)
                vr = np.fromiter(r_d.keys(), dtype=float)
                pr = np.fromiter(r_d.values(), dtype=float)
                pr = pr / pr.sum()

                pt = self.get_transition_estimate(observation, a)

                p = np.zeros(len(pr) * self.nS)
                v = np.zeros(len(pr) * self.nS)
                k = 0
                for i in range(self.nS):
                    for j in range(len(pr)):
                        p[k] = pt[i] * pr[j]
                        v[k] = phi[i] + vr[j]
                        k += 1

                delta = v - mu
                h = lambda x: -np.sum(p * np.log(upper_bound - delta * x))

                res = minimize_scalar(h, bounds=(0, u), method="bounded")
                x = -res.fun
                self.index[a] = n * x + np.log(max(n, 1))

    def play(self, observation):
        """Select an action: forced exploration or IMED index minimization."""
        valid_actions = self.get_available_actions(observation)

        if not self.get_all_selected(observation):
            pulls = np.array(
                [self.get_pull_count(observation, a) for a in valid_actions], dtype=int
            )
            idx = randamin(pulls)
            action = valid_actions[idx]
        else:
            self.value_iteration(observation)
            self.multinomial_imed(observation)
            action = randamin(self.index, i=valid_actions)

        return action

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by subclasses
    # ------------------------------------------------------------------

    def _init_statistics(self):
        raise NotImplementedError

    def _reset_statistics(self, observation):
        raise NotImplementedError

    def get_phi(self, observation):
        raise NotImplementedError

    def set_phi(self, observation, phi):
        raise NotImplementedError

    def get_all_selected(self, observation):
        raise NotImplementedError

    def set_all_selected(self, observation, value):
        raise NotImplementedError

    def get_pull_count(self, observation, action):
        raise NotImplementedError

    def get_reward_estimate(self, observation, action):
        raise NotImplementedError

    def get_transition_estimate(self, observation, action):
        raise NotImplementedError

    def get_reward_distribution(self, observation, action):
        raise NotImplementedError

    def get_reward_from_state_action(self, observation, state, action):
        raise NotImplementedError

    def get_transition_from_state_action(self, observation, state, action):
        raise NotImplementedError

    def update_statistics(self, observation, action, reward, next_observation):
        raise NotImplementedError

class GlobalIMEDRL(ContextualIMEDRL):
    """Context-aware observations, context-agnostic learning.
 
    Pools all samples across contexts and maintains global estimates
    indexed only by (s, a), with a single value function phi(s).
 
    Parameters
    ----------
    nbr_states : int
        Number of states in the MDP.
    nbr_actions : int
        Number of actions.
    nbr_contexts : int
        Number of discrete contexts.
    skeleton : dict, optional
        Mapping state -> array of admissible action indices.
    max_iter : int
        Maximum iterations for value iteration.
    epsilon : float
        Convergence threshold for value iteration.
    max_reward : float
        Upper bound on single-step rewards.
    """
 
    def __init__(
        self,
        nbr_states,
        nbr_actions,
        nbr_contexts,
        skeleton=None,
        max_iter=3000,
        epsilon=1e-3,
        max_reward=1,
    ):
        super().__init__(
            nbr_states=nbr_states,
            nbr_actions=nbr_actions,
            nbr_contexts=nbr_contexts,
            skeleton=skeleton,
            learning_scope="global",
            name="GlobalIMED-RL",
            max_iter=max_iter,
            epsilon=epsilon,
            max_reward=max_reward,
        )
 
    def _init_statistics(self):
        self.state_action_pulls = np.zeros((self.nS, self.nA), dtype=int)
        self.state_visits = np.zeros(self.nS, dtype=int)
        self.rewards = np.full((self.nS, self.nA), 0.5)
        self.transitions = np.full((self.nS, self.nA, self.nS), 1.0 / self.nS)
        self._all_selected = np.zeros(self.nS, dtype=bool)
        self._phi = np.zeros(self.nS)
        self.rewards_distributions = {
            s: {a: {1: 0, 0.5: 1} for a in range(self.nA)}
            for s in range(self.nS)
        }
 
    def _reset_statistics(self, observation):
        self._init_statistics()
 
    def get_phi(self, observation):
        return self._phi
 
    def set_phi(self, observation, phi):
        self._phi = phi
 
    def get_all_selected(self, observation):
        s = self.get_state(observation)
        return self._all_selected[s]
 
    def set_all_selected(self, observation, value):
        s = self.get_state(observation)
        self._all_selected[s] = value
 
    def get_pull_count(self, observation, action):
        s = self.get_state(observation)
        return self.state_action_pulls[s, action]
 
    def get_reward_estimate(self, observation, action):
        s = self.get_state(observation)
        return self.rewards[s, action]
 
    def get_transition_estimate(self, observation, action):
        s = self.get_state(observation)
        return self.transitions[s, action]
 
    def get_reward_distribution(self, observation, action):
        s = self.get_state(observation)
        return self.rewards_distributions[s][action]
 
    def get_reward_from_state_action(self, observation, state, action):
        return self.rewards[state, action]
 
    def get_transition_from_state_action(self, observation, state, action):
        return self.transitions[state, action]
 
    def update_statistics(self, observation, action, reward, next_observation):
        s = self.get_state(observation)
        next_s = self.get_state(next_observation)
 
        na = self.state_action_pulls[s, action]
        r = self.rewards[s, action]
        p = self.transitions[s, action]
 
        self.state_action_pulls[s, action] = na + 1
        self.state_visits[s] = self.state_visits[s] + 1
        self.rewards[s, action] = ((na + 1) * r + reward) / (na + 2)
        self.transitions[s, action] = ((na + 1) * p + self.dirac[next_s]) / (na + 2)
 
        rd = self.rewards_distributions[s][action]
        if reward in rd:
            rd[reward] += 1
        else:
            rd[reward] = 1

class SemiLocalIMEDRL(ContextualIMEDRL):
    """Context-dependent reward learning, context-independent transitions.
 
    Maintains separate empirical reward statistics for each (x, s, a)
    triplet, while keeping transition estimates global in (s, a).
    Computes a context-dependent value function phi_x(s).
 
    Parameters
    ----------
    nbr_states : int
        Number of states in the MDP.
    nbr_actions : int
        Number of actions.
    nbr_contexts : int
        Number of discrete contexts.
    skeleton : dict, optional
        Mapping state -> array of admissible action indices.
    max_iter : int
        Maximum iterations for value iteration.
    epsilon : float
        Convergence threshold for value iteration.
    max_reward : float
        Upper bound on single-step rewards.
    """
 
    def __init__(
        self,
        nbr_states,
        nbr_actions,
        nbr_contexts,
        skeleton=None,
        max_iter=3000,
        epsilon=1e-3,
        max_reward=1,
    ):
        super().__init__(
            nbr_states=nbr_states,
            nbr_actions=nbr_actions,
            nbr_contexts=nbr_contexts,
            skeleton=skeleton,
            learning_scope="semi-local",
            name="SemiLocalIMED-RL",
            max_iter=max_iter,
            epsilon=epsilon,
            max_reward=max_reward,
        )
 
    def _init_statistics(self):
        # Local: indexed by (c, s, a)
        self.state_action_pulls = np.zeros((self.nC, self.nS, self.nA), dtype=int)
        self.state_visits = np.zeros((self.nC, self.nS), dtype=int)
        self.rewards = np.full((self.nC, self.nS, self.nA), 0.5)
        self._all_selected = np.zeros((self.nC, self.nS), dtype=bool)
        self._phi = np.zeros((self.nC, self.nS))
        self.rewards_distributions = {
            c: {s: {a: {1: 0, 0.5: 1} for a in range(self.nA)} for s in range(self.nS)}
            for c in range(self.nC)
        }
 
        # Global: indexed by (s, a)
        self.transition_pulls = np.zeros((self.nS, self.nA), dtype=int)
        self.transitions = np.full((self.nS, self.nA, self.nS), 1.0 / self.nS)
 
    def _reset_statistics(self, observation):
        self._init_statistics()
 
    def get_phi(self, observation):
        c = self.get_context(observation)
        return self._phi[c]
 
    def set_phi(self, observation, phi):
        c = self.get_context(observation)
        self._phi[c] = phi
 
    def get_all_selected(self, observation):
        c = self.get_context(observation)
        s = self.get_state(observation)
        return self._all_selected[c, s]
 
    def set_all_selected(self, observation, value):
        c = self.get_context(observation)
        s = self.get_state(observation)
        self._all_selected[c, s] = value
 
    def get_pull_count(self, observation, action):
        c = self.get_context(observation)
        s = self.get_state(observation)
        return self.state_action_pulls[c, s, action]
 
    def get_reward_estimate(self, observation, action):
        c = self.get_context(observation)
        s = self.get_state(observation)
        return self.rewards[c, s, action]
 
    def get_transition_estimate(self, observation, action):
        s = self.get_state(observation)
        return self.transitions[s, action]
 
    def get_reward_distribution(self, observation, action):
        c = self.get_context(observation)
        s = self.get_state(observation)
        return self.rewards_distributions[c][s][action]
 
    def get_reward_from_state_action(self, observation, state, action):
        c = self.get_context(observation)
        return self.rewards[c, state, action]
 
    def get_transition_from_state_action(self, observation, state, action):
        return self.transitions[state, action]
 
    def update_statistics(self, observation, action, reward, next_observation):
        c = self.get_context(observation)
        s = self.get_state(observation)
        next_s = self.get_state(next_observation)
 
        # Local reward update
        na_r = self.state_action_pulls[c, s, action]
        r = self.rewards[c, s, action]
 
        self.state_action_pulls[c, s, action] = na_r + 1
        self.state_visits[c, s] = self.state_visits[c, s] + 1
        self.rewards[c, s, action] = ((na_r + 1) * r + reward) / (na_r + 2)
 
        rd = self.rewards_distributions[c][s][action]
        if reward in rd:
            rd[reward] += 1
        else:
            rd[reward] = 1
 
        # Global transition update
        na_t = self.transition_pulls[s, action]
        p = self.transitions[s, action]
 
        self.transition_pulls[s, action] = na_t + 1
        self.transitions[s, action] = ((na_t + 1) * p + self.dirac[next_s]) / (na_t + 2)
 