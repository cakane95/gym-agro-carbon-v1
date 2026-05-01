/**
* Name: EcoSysMLBehavior
* Defines the dynamic aspects of a socio-ecological system. 
* Provides the elements needed to model the behavior of actors within an SES,
* describing how they interact with their environment and make decisions.
* Author: Cheikhou Akhmed KANE
* Tags: 
*/

model EcoSysMLBehavior

import "EcoSysMLStructure.gaml"

/*
 * The Activity concept represents actions undertaken by an actor within
 * the socio-ecological system. Each Activity is performed by an Actor 
 * and consists of one or more Tasks.
 */
species Activity {
	string name;
    list<Task> ownedTasks;
}

/*
 * The Task concept represents the smallest unit of work within an Activity.
 * Abstract parent for StepTask and CutTask.
 */
species Task {
	string name;
}

/*
 * StepTask handles the core MDP dynamics:
 * 1. Compute the reward for (s_current, a_real)
 * 2. Compute the state transition (aging logic)
 *
 * Reward:
 *   - agnostic:   R(s,a)   ~ N(base_means[a] + age_bonus(s), noise)
 *   - contextual: R(c,s,a) ~ N(base_means[a] * context_scales[c] + age_bonus(s), noise)
 *
 * Transition:
 *   - s=0 and a=trigger_action -> s=1 (RNA initiated)
 *   - s>0 -> s=min(s+1, nS-1) (automatic aging)
 *   - otherwise -> s=0 (no tree, no planting)
 */
species StepTask parent: Task {

    /*
     * Execute one MDP step on a parcel.
     *
     * @param cell: the target Parcel
     * @param action_id: the action actually executed (a_real, not a_recommended)
     * @param base_means: list of base reward means per action
     * @param context_scales: list of reward multipliers per context
     * @param r_is_contextual: if true, reward depends on cell.soil_type_id
     * @param noise: standard deviation of reward gaussian
     * @param age_bonus_max: maximum age bonus at maturity
     * @param growth_rate: controls convexity of age bonus curve
     * @param nS: number of states (max age = nS - 1)
     * @param trigger_action: action id that initiates RNA
     *
     * @return the sampled reward
     */
    action execute_on(Parcel cell, int action_id,
                      list<float> base_means,
                      list<float> context_scales,
                      list<float> action_bonus_scales,
                      bool r_is_contextual,
                      float noise,
                      float age_bonus_max,
                      float growth_rate,
                      int nS,
                      int trigger_action) type: float {

        int s_current <- cell.new_tree_age;

        // --- 1. Compute reward R(s_current, a_real) or R(c, s_current, a_real) ---
		float base_mean <- base_means[action_id];
		float age_bonus <- compute_age_bonus(s_current, nS, age_bonus_max, growth_rate);
		float action_bonus_scale <- action_bonus_scales[action_id];
		
		float reward_mean;
		if (r_is_contextual) {
		    float scale <- context_scales[cell.soil_type_id];
		    reward_mean <- base_mean * scale + age_bonus * action_bonus_scale;
		} else {
		    reward_mean <- base_mean + age_bonus * action_bonus_scale;
		}
		
		float reward <- gauss(reward_mean, noise);

        // --- 2. State transition ---
        int s_next;
        if (s_current = 0) {
            if (action_id = trigger_action) {
                s_next <- 1;  // RNA initiated: protect the young shoot
                create Tree { location <- cell.location; }
            } else {
                s_next <- 0;  // No tree, no planting
            }
        } else {
            s_next <- min(s_current + 1, nS - 1);  // Automatic aging
        }

        cell.new_tree_age <- s_next;

        return reward;
    }

    /*
     * Convex exponential age bonus, matching Python's _age_bonus().
     * - At s=0: returns 0.0 exactly
     * - Grows slowly then accelerates
     * - At s=nS-1: returns age_bonus_max exactly
     */
    action compute_age_bonus(int s, int nS, float age_bonus_max, float growth_rate) type: float {
        if (nS <= 1) { return 0.0; }

        float t <- float(s) / float(nS - 1);

        return age_bonus_max
            * (exp(growth_rate * t) - 1.0)
            / (exp(growth_rate) - 1.0);
    }
}

/*
 * CutTask handles the stochastic risk of tree destruction.
 * Applies after StepTask: if the tree survives, s_next stands.
 * If cut, the tree is destroyed and the parcel returns to s=0.
 *
 * Later, this logic can be driven by a Bandit agent with 
 * its own decision process instead of a simple coin flip.
 */
species CutTask parent: Task {
	
	/*
	 * Evaluate and apply the risk of tree cutting.
	 *
	 * @param cell: the target Parcel
	 * @param p_cut: probability of the tree being cut down
	 *
	 * @return true if the tree was cut, false otherwise
	 */
	action execute_on(Parcel cell, float p_cut) type: bool {
		
		// Only applies if a tree exists (s > 0)
		if (cell.new_tree_age > 0 and p_cut > 0.0) {
			if (flip(p_cut)) {
				// Tree is cut down — destroy visual and reset state
				Tree tree_on_cell <- Tree first_with (each.location = cell.location);
				if (tree_on_cell != nil) {
					ask tree_on_cell { do die; }
				}
				cell.new_tree_age <- 0;
				return true;
			}
		}
		return false;
	}
}
