/**
* Name: main
* Agrocarbon RNA environment for contextual RL.
* General list-based interface for GAMA headless control.
*
* Python does not call GAML actions directly.
* Python only sets variables using expressions, then steps GAMA.
*
* Author: Cheikhou Akhmed KANE
* Tags: reinforcement-learning, agroforestry, RNA, contextual-MDP
*/

model main

import "EcoSysMLStructure.gaml"
import "EcoSysMLBehavior.gaml"

global {
	
	float forced_seed <- -1.0;

    // =============================================
    // --- Communication ---
    // =============================================

    int gama_server_port <- 0;

    // =============================================
    // --- MDP Parameters ---
    // =============================================

    int nS <- 4;
    int nA <- 4;
    int nC <- 3;
    int trigger_action <- 2;
    float p_cut <- 0.0;

    // =============================================
    // --- Reward Parameters ---
    // =============================================

    list<float> base_means <- [0.4, 0.6, 0.8, 1.0];
    list<float> action_bonus_scales <- [0.10, 0.40, 0.20, 1.00];
    float reward_noise <- 0.05;
    float age_bonus_max <- 0.36;
    float growth_rate <- 3.0;
	bool r_is_contextual <- false;

	// =============================================
	// --- Dynamic Parameters ---
	// =============================================
	
	bool p_is_contextual <- false;
	list<float> context_cut_scales <- [];
	
    // =============================================
    // --- Context Distribution ---
    // =============================================

    bool c_is_static <- true;
    list<float> context_dist <- [0.4, 0.4, 0.2];
    list<float> context_scales <- [1.0, 1.0, 1.0];
    
    // ============================================
    // --- BDI Farmer Agents ---
    // ============================================
    
    int farmer_household_size <- 1;
	float farmer_tree_knowledge <- 1.0;
	float farmer_base_compliance <- 1.0;
	float farmer_food_pressure_penalty <- 0.0;
	float farmer_tree_knowledge_bonus <- 0.0;
	int farmer_fallback_action <- 3;
    
    // =============================================
    // --- GUI / Testing Parameters ---
    // =============================================

    int default_action <- 0;
    int step_count <- 0;
    int max_gui_steps <- 10;

    // In gym_env, Python must reset before simulation starts.
    // In test_env, we can allow the GUI to start directly.
    bool wait_for_python_reset <- true;

    // =============================================
    // --- Python Reset Handshake Variables ---
    // =============================================

    list<int> pending_contexts <- [];
    list<int> pending_states <- [];
    bool reset_requested <- false;

    // Normal simulation steps are blocked until reset is applied.
    bool ready_for_step <- false;

    // Prevents reset and simulation_cycle from executing in the same GAMA cycle.
    int last_reset_cycle <- -1;

    // =============================================
    // --- Agent References ---
    // =============================================

    StepTask step_engine;
    CutTask cut_engine;
    GymAgent gym_interface;
    Farmer farmer;

    // =============================================
    // --- Initialization ---
    // =============================================

    init {
    	
    	if (forced_seed >= 0.0) {
        	seed <- forced_seed;
    	}

        if (length(base_means) != nA) {
            error "base_means length must match nA.";
        }
		
		if (r_is_contextual and length(context_scales) != nC) {
		    error "context_scales length must match nC when r_is_contextual = true.";
		}

        // --- Create engine agents ---

        create StepTask number: 1 returns: step_agents {
            name <- "Step_Engine";
        }
        step_engine <- first(step_agents);

        create CutTask number: 1 returns: cut_agents {
            name <- "Cut_Engine";
        }
        cut_engine <- first(cut_agents);

        // --- Create GymAgent interface ---

        create GymAgent number: 1 returns: gym_agents {
            name <- "Gym_Interface";
        }
        gym_interface <- first(gym_agents);

        gym_interface.action_space <- [
            "type"::"MultiDiscrete",
            "nvec"::(Parcel collect nA)
        ];

        gym_interface.observation_space <- [
            "type"::"Box",
            "low"::0,
            "high"::max(nS - 1, nC - 1),
            "shape"::[length(Parcel), 2],
            "dtype"::"int"
        ];

        gym_interface.next_action <- Parcel collect default_action;

        // --- Create Farmer ---

        create Farmer number: 1 returns: farmers {
		    id <- "Farmer_1";
		    household_size <- world.farmer_household_size;
		    tree_knowledge <- world.farmer_tree_knowledge;
		    base_compliance <- world.farmer_base_compliance;
		    food_pressure_penalty <- world.farmer_food_pressure_penalty;
		    tree_knowledge_bonus <- world.farmer_tree_knowledge_bonus;
		    fallback_action <- world.farmer_fallback_action;
		}
        farmer <- first(farmers);

        // --- Initialize parcels with internal defaults ---

        ask Parcel {
            self.soil_type_id <- world.sample_context();
            self.new_tree_age <- 0;
            self.C_input <- 0.0;
            self.yield <- 0.0;

            world.farmer.my_parcel <- self;
        }

        // In GUI test mode, allow the model to run without Python reset.
        if (!wait_for_python_reset) {
            ready_for_step <- true;
        }

        list<unknown> init_state <- build_observation();

        map<string, unknown> init_info <- [
            "message"::"environment_initialized",
            "n_parcels"::length(Parcel),
            "ready_for_step"::ready_for_step,
            "wait_for_python_reset"::wait_for_python_reset
        ];

        ask gym_interface {
            do update_data obs: init_state r: 0.0 term: false trunc: false i: init_info;
        }

        write "GAMA: Environment initialized.";
        write "GAMA: Number of parcels = " + length(Parcel);
        write "GAMA: Initial state = " + init_state;
        write "GAMA: ready_for_step = " + ready_for_step;
    }

    // =============================================
    // --- Context Sampling ---
    // =============================================

    action sample_context type: int {
        float rand <- rnd(1.0);
        float cumul <- 0.0;

        loop i from: 0 to: length(context_dist) - 1 {
            cumul <- cumul + context_dist[i];

            if (rand <= cumul) {
                return i;
            }
        }

        return length(context_dist) - 1;
    }

    // =============================================
    // --- Build Territorial Observation ---
    // =============================================

    action build_observation type: list<unknown> {
        return Parcel collect ([each.soil_type_id, each.new_tree_age]);
    }

    // =============================================
    // --- Build Parcel-Level Info ---
    // =============================================

    action build_parcel_info(
	    list<int> actions_recommended,
	    list<int> actions_executed,
	    list<float> rewards,
	    list<bool> cut_flags
	) type: list<unknown> {
	
	    list<unknown> parcel_info <- [];
	    int idx <- 0;
	
	    loop cell over: Parcel {
	
	        map<string, unknown> one_info <- [
	            "parcel_index"::idx,
	            "context"::cell.soil_type_id,
	            "tree_age"::cell.new_tree_age,
	            "action_recommended"::actions_recommended[idx],
	            "action_executed"::actions_executed[idx],
	            "complied"::(actions_recommended[idx] = actions_executed[idx]),
	            "reward"::rewards[idx],
	            "was_cut"::cut_flags[idx],
	            "yield"::cell.yield,
	            "C_input"::cell.C_input
	        ];
	
	        parcel_info <- parcel_info + [one_info];
	        idx <- idx + 1;
	    }
	
	    return parcel_info;
	}

    // =============================================
    // --- Reset Reflex Triggered by Python ---
    // =============================================

    reflex apply_python_reset when: reset_requested {

        ready_for_step <- false;
        step_count <- 0;
        
        ask Tree {
        do die;
    	}

        int n_parcels <- length(Parcel);

        if (length(pending_contexts) != n_parcels) {
            error "apply_python_reset: pending_contexts length does not match number of parcels.";
        }

        if (length(pending_states) != n_parcels) {
            error "apply_python_reset: pending_states length does not match number of parcels.";
        }

        int idx <- 0;

        loop cell over: Parcel {

            int c <- pending_contexts[idx];
            int s <- pending_states[idx];

            if (c < 0) {
                c <- 0;
            }

            if (c >= nC) {
                c <- nC - 1;
            }

            if (s < 0) {
                s <- 0;
            }

            if (s >= nS) {
                s <- nS - 1;
            }

            cell.soil_type_id <- c;
            cell.new_tree_age <- s;
            cell.C_input <- 0.0;
            cell.yield <- 0.0;

            idx <- idx + 1;
        }

        list<unknown> reset_state <- build_observation();

        map<string, unknown> reset_info <- [
            "message"::"reset_from_python",
            "n_parcels"::n_parcels,
            "contexts"::pending_contexts,
            "states"::pending_states,
            "ready_for_step"::true,
            "cycle"::cycle
        ];

        ask gym_interface {
            do update_data obs: reset_state r: 0.0 term: false trunc: false i: reset_info;
        }

        reset_requested <- false;
        ready_for_step <- true;

        // Important: prevents simulation_cycle from running in this same cycle.
        last_reset_cycle <- cycle;

        write "GAMA: Python reset applied.";
        write "GAMA: Reset state = " + reset_state;
        write "GAMA: Reset cycle = " + cycle;
    }

    // =============================================
    // --- Main Simulation Cycle ---
    // =============================================

    reflex simulation_cycle when: ready_for_step and not reset_requested and cycle > last_reset_cycle {

        if (step_count >= max_gui_steps) {
            write "GAMA: reached max_gui_steps = " + max_gui_steps + ". Pausing simulation.";
            do pause;
        }

        list<int> recommended_actions <- list<int>(gym_interface.next_action);

        if (length(recommended_actions) = 0) {
            recommended_actions <- Parcel collect default_action;
        }

        list<int> actions_executed <- [];
        list<float> parcel_rewards <- [];
        list<bool> cut_flags <- [];

        float total_reward <- 0.0;

        int idx <- 0;

        loop cell over: Parcel {

            int a_recommended;

            if (idx < length(recommended_actions)) {
                a_recommended <- recommended_actions[idx];
            } else {
                a_recommended <- default_action;
            }

            int a_real <- a_recommended;

			ask farmer {
			    a_real <- decide_action(a_recommended);
			}

            if (a_real < 0) {
                a_real <- 0;
            }

            if (a_real >= nA) {
                a_real <- nA - 1;
            }

            float r <- 0.0;

            ask step_engine {
               r <- execute_on(
				    cell,
				    a_real,
				    world.base_means,
				    world.context_scales,
				    world.action_bonus_scales,
				    world.r_is_contextual,
				    world.reward_noise,
				    world.age_bonus_max,
				    world.growth_rate,
				    world.nS,
				    world.trigger_action
				);
            }

            bool was_cut <- false;

            ask cut_engine {
                was_cut <- execute_on(
				    cell,
				    world.p_cut,
				    world.p_is_contextual,
				    world.context_cut_scales
				);
            }
            

            actions_executed <- actions_executed + [a_real];
            parcel_rewards <- parcel_rewards + [r];
            cut_flags <- cut_flags + [was_cut];

            total_reward <- total_reward + r;
            
            if (!world.c_is_static) {
			    cell.soil_type_id <- sample_context();
			}

            idx <- idx + 1;
        }

        list<unknown> new_state <- build_observation();

        list<unknown> parcel_info <- build_parcel_info(
        	recommended_actions,
            actions_executed,
            parcel_rewards,
            cut_flags
        );

        map<string, unknown> step_info <- [
            "message"::"step_completed",
            "step"::step_count,
            "cycle"::cycle,
            "n_parcels"::length(Parcel),
            "actions_executed"::actions_executed,
            "actions_recommended"::recommended_actions,
			"compliance_probability"::farmer.compliance_probability,
			"farmer_complied"::farmer.complied,
			"household_size"::farmer.household_size,
			"tree_knowledge"::farmer.tree_knowledge,
            "parcel_rewards"::parcel_rewards,
            "cut_flags"::cut_flags,
            "parcel_info"::parcel_info
        ];

        ask gym_interface {
            do update_data obs: new_state r: total_reward term: false trunc: false i: step_info;
        }

        step_count <- step_count + 1;

        write "-----------------------------";
        write "Step " + step_count;
        write "Cycle " + cycle;
        write "Actions executed: " + actions_executed;
        write "Parcel rewards: " + parcel_rewards;
        write "Cut flags: " + cut_flags;
        write "Total reward: " + total_reward;
        write "New state: " + new_state;
        write "Gym data: " + gym_interface.data;
    }
}

// =============================================
// --- GymAgent: Communication Interface ---
// =============================================

species GymAgent {

    map<string, unknown> action_space;
    map<string, unknown> observation_space;

    unknown state;
    float reward <- 0.0;
    bool terminated <- false;
    bool truncated <- false;
    map<string, unknown> info <- [];

    unknown next_action <- [];
    map<string, unknown> data;

    action update_data(
        unknown obs,
        float r,
        bool term,
        bool trunc,
        map<string, unknown> i
    ) {
        state <- obs;
        reward <- r;
        terminated <- term;
        truncated <- trunc;
        info <- i;

        data <- [
            "State"::state,
            "Reward"::reward,
            "Terminated"::terminated,
            "Truncated"::truncated,
            "Info"::info
        ];
    }
}

// =============================================
// --- Experiment Definitions ---
// =============================================

experiment test_env type: gui {

    parameter "Number of States" var: nS <- 4;
    parameter "Number of Actions" var: nA <- 4;
    parameter "Number of Contexts" var: nC <- 3;
    parameter "Trigger Action" var: trigger_action <- 2;
    parameter "Cut Probability" var: p_cut <- 0.0;

    parameter "Base Means" var: base_means <- [0.4, 0.6, 0.8, 1.0];
    parameter "Age Bonus Max" var: age_bonus_max <- 0.36;
    parameter "Growth Rate" var: growth_rate <- 3.0;
    parameter "Reward Noise" var: reward_noise <- 0.05;

    parameter "Default Action" var: default_action <- 2;
    parameter "Max GUI Steps" var: max_gui_steps <- 10;

    // GUI mode runs without waiting for Python reset.
    parameter "Wait for Python Reset" var: wait_for_python_reset <- false;

    output {
        display main_display {
            grid Parcel border: #black;
            species Tree aspect: base;
        }
    }
}

experiment gym_env type: gui {
	
	parameter "Forced Seed" var: forced_seed <- -1.0;

    parameter "Server Port" var: gama_server_port;
    parameter "Action Bonus Scales" var: action_bonus_scales <- [0.10, 0.40, 0.20, 1.00];

    parameter "Number of States" var: nS;
    parameter "Number of Actions" var: nA;
    parameter "Number of Contexts" var: nC;
    parameter "Trigger Action" var: trigger_action;
    parameter "Cut Probability" var: p_cut;
    
    parameter "Context Is Static" var: c_is_static <- true;

    parameter "Reward Is Contextual" var: r_is_contextual <- false;
	parameter "Context Scales" var: context_scales <- [1.0, 1.0, 1.0];
    parameter "Base Means" var: base_means;
    parameter "Age Bonus Max" var: age_bonus_max;
    parameter "Growth Rate" var: growth_rate;
    parameter "Reward Noise" var: reward_noise;
    
    parameter "Farmer Household Size" var: farmer_household_size <- 1;
	parameter "Farmer Tree Knowledge" var: farmer_tree_knowledge <- 1.0;
	parameter "Farmer Base Compliance" var: farmer_base_compliance <- 1.0;
	parameter "Farmer Food Pressure Penalty" var: farmer_food_pressure_penalty <- 0.0;
	parameter "Farmer Tree Knowledge Bonus" var: farmer_tree_knowledge_bonus <- 0.0;
	parameter "Farmer Fallback Action" var: farmer_fallback_action <- 3;

    parameter "Default Action" var: default_action;
    parameter "Max GUI Steps" var: max_gui_steps;

    // Headless gym mode waits for Python-controlled reset.
    parameter "Wait for Python Reset" var: wait_for_python_reset <- true;

    output {
        display main_display {
            grid Parcel border: #black;
            species Tree aspect: base;
        }
    }
}