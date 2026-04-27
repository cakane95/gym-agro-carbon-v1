from statisticalrl_environments.MDPs_discrete.utils import Dirac
from scipy.stats import norm

from contextualMDP import ContextualDiscreteMDP

class AgroCarbonTreeMDP(ContextualDiscreteMDP):
    pass

    # Attributs
    """
    - M : tree maturity age such that nS = M + 1
    - reward_mode : "global" as default, "local" to know how to interpret R
    - mean_rewards
    - sigma
    - isd = Dirac(0)
    """

    # Methodes
    """
    - get_available_actions(state=None) -> A(s)
    - build_transition_kernel() -> P
    - build_reward_model() -> R
    - get_mean_reward(x, s, a)
    - get_action_mask()
    """