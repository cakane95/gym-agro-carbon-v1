#./src/contextual_stat_rl/environments/ContextualMDPs_discrete/factories/agrocarbon_factory.py

import math
import scipy.stats as stat

# ==========================================
# Helper Functions
# ==========================================

# ==========================================
# Dynamics
# ==========================================

def build_tree_transitions(nS, nA, trigger_action, p_cut=0.0):
    """
    Builds the transition matrix P[s][a] for the aging dynamics.
    
    Parameters:
    - nS: Total number of states (representing tree age, max age is nS - 1).
    - nA: Total number of actions.
    - trigger_action: The action that plants the tree (transition s=0 -> s=1).
    - p_cut: Probability (0.0 to 1.0) that the tree is cut down and returns to state 0.
    """
    P = {}
    for s in range(nS):
        P[s] = {}
        for a in range(nA):
            
            # 1. Determine the natural next state
            if s == 0:
                if a == trigger_action:
                    base_next_s = 1
                else:
                    base_next_s = 0
            else:
                base_next_s = min(s + 1, nS - 1)

            # 2. Apply stochasticity (risk of being cut)
            if p_cut > 0.0 and base_next_s != 0:
                P[s][a] = [
                    (1.0 - p_cut, base_next_s, False),
                    (p_cut, 0, False)
                ]
            else:
                P[s][a] = [(1.0, base_next_s, False)]
                
    return P

def build_context_p_cut_scales(nC, gap=0.05, reference_context=0):
    """
    Build multiplicative context scales for cutting risk.

    Contexts are assumed to be ordered from less productive to more productive.
    More productive contexts receive slightly higher cutting-risk scales.
    """
    if nC <= 0:
        raise ValueError("nC must be positive.")

    if gap < 0:
        raise ValueError("context_p_cut_scale_gap must be non-negative.")

    if reference_context < 0 or reference_context >= nC:
        raise ValueError(
            f"reference_context must be in [0, {nC - 1}], got {reference_context}"
        )

    scales = [
        1.0 + gap * (c - reference_context)
        for c in range(nC)
    ]

    if any(scale < 0.0 for scale in scales):
        raise ValueError(
            f"Invalid context_p_cut_scales={scales}. "
            "Decrease gap or change reference_context."
        )

    return scales

def build_contextual_p_cut(p_cut, context_p_cut_scales):
    """
    Convert baseline p_cut and context scales into valid context-specific
    cutting probabilities.
    """
    if p_cut < 0.0 or p_cut > 1.0:
        raise ValueError(f"p_cut must be in [0, 1], got {p_cut}")

    contextual_p_cut = []

    for scale in context_p_cut_scales:
        if scale < 0.0:
            raise ValueError(
                f"context_p_cut_scales must be non-negative, got {scale}"
            )

        p_c = p_cut * scale

        if p_c < 0.0 or p_c > 1.0:
            raise ValueError(
                f"Invalid contextual cutting probability p_cut^c={p_c}. "
                f"Got p_cut={p_cut}, scale={scale}. "
                "Decrease p_cut or context_p_cut_scale_gap."
            )

        contextual_p_cut.append(p_c)

    return contextual_p_cut

def build_contextual_tree_transitions(
    nS,
    nA,
    nC,
    trigger_action,
    p_cut=0.0,
    context_p_cut_scale_gap=0.05,
    reference_context=0,
):
    """
    Builds contextual transition matrices P[c][s][a] for tree-aging dynamics.

    The biological aging rule is shared across contexts, but the tree cutting
    risk is context-dependent. Contexts are assumed to be ordered from less
    productive to more productive. By default, more productive contexts have
    slightly higher cutting risk because their parkland resources may be more
    contested.

    Parameters
    ----------
    nS : int
        Number of states, representing tree age from 0 to nS - 1.
    nA : int
        Number of actions.
    nC : int
        Number of contexts.
    trigger_action : int
        Action that initiates ANR/tree protection.
    p_cut : float, default=0.0
        Baseline cutting probability.
    context_p_cut_scale_gap : float, default=0.05
        Increment in cutting-risk scale across contexts.
    reference_context : int, default=0
        Context index with scale 1.0.

    Returns
    -------
    dict
        Contextual transition dictionary P[c][s][a].
    """
    context_p_cut_scales = build_context_p_cut_scales(
        nC=nC,
        gap=context_p_cut_scale_gap,
        reference_context=reference_context,
    )

    contextual_p_cut = build_contextual_p_cut(
        p_cut=p_cut,
        context_p_cut_scales=context_p_cut_scales,
    )

    P = {}

    for c in range(nC):
        P[c] = build_tree_transitions(
            nS=nS,
            nA=nA,
            trigger_action=trigger_action,
            p_cut=contextual_p_cut[c],
        )

    return P

# ==========================================
# Skeletons
# ==========================================

def build_action_names(nA):
    """Returns a list of action names."""
    if nA == 4:
        return ["fallow", "fert_fallow", "tree", "baseline"]
    return [f"action_{i}" for i in range(nA)]

def build_tree_skeleton(nS, nA, trigger_action, trigger_state=0):
    """Builds the action mask: trigger_action only available at trigger_state."""
    skeleton = {}
    all_actions = list(range(nA))
    actions_without_trigger = [a for a in all_actions if a != trigger_action]
    
    for s in range(nS):
        if s == trigger_state:
            skeleton[s] = all_actions.copy()
        else:
            skeleton[s] = actions_without_trigger.copy()
    return skeleton

# ==========================================
# Rewards
# ==========================================

def _age_bonus(s, nS, age_bonus_max=0.40, growth_rate=3.0):
    """
    Convex exponential tree-age bonus.

    The bonus is a normalized proxy for the delayed production-carbon benefit
    of tree growth. It is not difficulty-dependent: easy and hard scenarios
    differ through base reward gaps, context gaps, observation noise, and
    transition stochasticity, not through the assumed mature-tree benefit.

    - At s=0: bonus = 0.0 exactly
    - Grows slowly for young trees, then accelerates toward maturity
    - At s=nS-1: bonus = age_bonus_max exactly
    - growth_rate controls the convexity of the curve
    """
    if nS <= 1:
        return 0.0

    t = s / (nS - 1)

    return (
        age_bonus_max
        * (math.exp(growth_rate * t) - 1)
        / (math.exp(growth_rate) - 1)
    )

def _build_base_means(nA, difficulty):
    """
    Generate normalized base reward means for each action.

    The last action is always the baseline/conventional cropping action,
    fixed at 1.0. Other actions receive non-zero values because the benchmark
    approximates a mixed production-carbon objective rather than crop yield alone.

    For nA=4, the intended action order is:
    0 = fallow
    1 = fertilized fallow
    2 = tree/RNA
    3 = baseline/conventional cropping
    """
    if difficulty == "easy":
        gap = 0.20
    else:
        gap = 0.10

    baseline = 1.0

    if nA <= 1:
        return [baseline]

    # For nA actions, keep baseline as the last action and assign
    # evenly spaced lower means to the previous nA-1 actions.
    return [
        baseline - gap * (nA - 1 - i)
        for i in range(nA - 1)
    ] + [baseline]

def _build_context_scales(nC, difficulty):
    """
    Generate per-context productivity multipliers.

    Contexts represent stylized soil or parcel conditions. The best context
    is normalized to 1.0, while less favorable contexts slightly reduce the
    immediate reward potential. For instance, this can be interpreted as a
    coarse abstraction of differences between poorer sandy soils and more
    fertile soils.

    The easy setting uses larger context gaps; the hard setting uses tighter
    context gaps.
    """
    if difficulty == "easy":
        gap = 0.10
    else:
        gap = 0.05

    best = 1.0

    if nC <= 1:
        return [best]

    return [
        best - gap * (nC - 1 - c)
        for c in range(nC)
    ]

def build_agnostic_reward_matrix(nS, nA, nC, difficulty="easy"):
    """R[s][a] — same reward regardless of context."""
    base_means = _build_base_means(nA, difficulty)
    action_bonus_scales = _build_action_bonus_scales(nA)

    # Fixed agronomic delayed-tree effect.
    # Difficulty should affect statistical separability, not the assumed mature-tree benefit.
    age_bonus_max = 0.36
    growth_rate = 3.0

    # Difficulty controls observation noise.
    noise = 0.05 if difficulty == "easy" else 0.15

    R = {}
    for s in range(nS):
        R[s] = {}
        bonus = _age_bonus(s, nS, age_bonus_max, growth_rate)

        for a in range(nA):
            mean = base_means[a] + bonus * action_bonus_scales[a]
            R[s][a] = stat.norm(mean, noise)

    return R

def _build_action_bonus_scales(nA):
    """
    Generate per-action multipliers for the tree-age bonus.

    The tree-age bonus represents the delayed benefit of a mature tree.
    This benefit is action-dependent: it is strongest for cropping under
    tree influence and weaker for restorative or tree-protection actions.

    Action order:
    0 = fallow
    1 = fertilized fallow
    2 = tree/RNA
    3 = baseline/conventional cropping

    The scale is kept fixed across difficulty settings because it represents
    an agronomic mechanism rather than statistical difficulty.
    """
    default_scales = [0.10, 0.40, 0.20, 1.00]

    if nA <= len(default_scales):
        # Always keep the last action as the baseline if nA < 4
        return default_scales[-nA:]

    return default_scales + [0.50] * (nA - len(default_scales))

def build_contextual_reward_matrix(nS, nA, nC, difficulty="easy"):
    """R[c][s][a] — rewards vary by context."""
    base_means = _build_base_means(nA, difficulty)
    context_scales = _build_context_scales(nC, difficulty)
    action_bonus_scales = _build_action_bonus_scales(nA)

    # Fixed agronomic delayed-tree effect.
    # Difficulty should affect statistical separability, not the assumed mature-tree benefit.
    age_bonus_max = 0.36
    growth_rate = 3.0

    # Difficulty controls observation noise.
    noise = 0.05 if difficulty == "easy" else 0.15

    R = {}
    for c in range(nC):
        R[c] = {}

        for s in range(nS):
            R[c][s] = {}
            bonus = _age_bonus(s, nS, age_bonus_max, growth_rate)

            for a in range(nA):
                mean = (
                    base_means[a] * context_scales[c]
                    + bonus * action_bonus_scales[a]
                )
                R[c][s][a] = stat.norm(mean, noise)

    return R

# ==========================================
# Initial Distributions
# ==========================================

def build_initial_state_dist(nS, nC, start_state=0):
    """Builds the context-dependent initial state distributions mu0[c]."""
    mu0 = {}
    for c in range(nC):
        dist = [0.0] * nS
        safe_start = min(start_state, nS - 1)
        dist[safe_start] = 1.0
        mu0[c] = dist
    return mu0

def build_context_dist(nC):
    """Builds the probability distribution over contexts (nu)."""
    if nC == 3:
        return [0.4, 0.4, 0.2]
    return [1.0 / nC] * nC

# ==========================================
# Environment Config Builder
# ==========================================

def build_agnostic_agrocarbon_config(nS=4, nA=4, nC=3, trigger_action=2, 
                                     p_cut=0.0, difficulty="easy",
                                     c_is_static=True):
    return {
        "nS": nS,
        "nA": nA,
        "nC": nC, # <-- not used
        "P": build_tree_transitions(nS, nA, trigger_action, p_cut),
        "R": build_agnostic_reward_matrix(nS, nA, nC, difficulty), # <-- Uses Agnostic R
        "mu0": build_initial_state_dist(nS, nC, start_state=0),
        "nu": build_context_dist(nC),
        "skeleton": build_tree_skeleton(nS, nA, trigger_action),
        "c_is_static": c_is_static,
        "p_is_contextual": False,
        "r_is_contextual": False,
        "nameActions": build_action_names(nA),
        "seed": 123
    }

def build_reward_contextual_agrocarbon_config(nS=4, nA=4, nC=3, trigger_action=2, 
                                              p_cut=0.0, difficulty="easy",
                                              c_is_static=True,):
    return {
        "nS": nS,
        "nA": nA,
        "nC": nC,
        "P": build_tree_transitions(nS, nA, trigger_action, p_cut),
        "R": build_contextual_reward_matrix(nS, nA, nC, difficulty), # <-- Uses Contextual R
        "mu0": build_initial_state_dist(nS, nC, start_state=0),
        "nu": build_context_dist(nC),
        "skeleton": build_tree_skeleton(nS, nA, trigger_action),
        "c_is_static": c_is_static,
        "p_is_contextual": False,
        "r_is_contextual": True, # <-- Flag set to True
        "nameActions": build_action_names(nA),
        "seed": 123
    }

def build_fully_contextual_agrocarbon_config(nS=8, nA=4, nC=3, trigger_action=2,
                                             p_cut=0.0, difficulty="easy",
                                             c_is_static=True,
                                context_p_cut_scale_gap=0.05, reference_context=0):
    return {
        "nS": nS,
        "nA": nA,
        "nC": nC,
        "P": build_contextual_tree_transitions(
            nS=nS,
            nA=nA,
            nC=nC,
            trigger_action=trigger_action,
            p_cut=p_cut,
            context_p_cut_scale_gap=context_p_cut_scale_gap,
            reference_context=reference_context,
        ), # <-- Uses Contextual P
        "R": build_contextual_reward_matrix(nS, nA, nC, difficulty), # <-- Uses Contextual R
        "mu0": build_initial_state_dist(nS, nC, start_state=0),
        "nu": build_context_dist(nC),
        "skeleton": build_tree_skeleton(nS, nA, trigger_action),
        "c_is_static": c_is_static,
        "p_is_contextual": True,
        "r_is_contextual": True,
        "nameActions": build_action_names(nA),
        "seed": 123,
    }