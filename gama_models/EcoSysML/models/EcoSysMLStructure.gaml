/**
* Name: EcoSysMLStructure
* Defines the static elements of a socio-ecological system. 
* Author: Cheikhou Akhmed KANE
* Tags: 
*/

model EcoSysMLStructure

global {
    int W <- 1;
    int H <- 1;
}

/* 
 * The Actor refers to human agents that perform activities. It is
 * linked with Resource. It is assigned goals via the concept of Goal.
 */ 
species Actor {
	string id;
	list<Resource> ownedResources;
}

/*
 * Farmer agent with a minimal BDI architecture.
 *
 * The farmer receives an RL recommendation but may override it depending on:
 * - food pressure, proxied by household size;
 * - knowledge of tree benefits.
 *
 * The BDI part is intentionally simple:
 * - beliefs represent household pressure and knowledge of tree benefits;
 * - the action decide_action computes compliance from these beliefs;
 * - the returned action is the action actually executed in the environment.
 */
species Farmer parent: Actor control: simple_bdi {

    Parcel my_parcel;

    int household_size <- 1;
	float tree_knowledge <- 1.0;
	float base_compliance <- 1.0;
	float food_pressure_penalty <- 0.0;
	float tree_knowledge_bonus <- 0.0;
	int fallback_action <- 3;

    float compliance_probability <- 1.0;
    bool complied <- true;

    // Beliefs
    predicate food_pressure <- new_predicate("food_pressure");
    predicate tree_benefit <- new_predicate("tree_benefit");

    init {
        do update_beliefs;
    }

    /*
     * Update simple beliefs from farmer attributes.
     */
    action update_beliefs {

        if (household_size > 5) {
            if (!has_belief(food_pressure)) {
                do add_belief(food_pressure);
            }
        } else {
            if (has_belief(food_pressure)) {
                do remove_belief(food_pressure);
            }
        }

        if (tree_knowledge > 0.7) {
            if (!has_belief(tree_benefit)) {
                do add_belief(tree_benefit);
            }
        } else {
            if (has_belief(tree_benefit)) {
                do remove_belief(tree_benefit);
            }
        }
    }

    /*
     * Decide the real action executed by the farmer.
     *
     * If the recommendation is baseline, the farmer accepts it.
     * Otherwise, compliance depends on BDI beliefs.
     */
    action decide_action(int a_recommended) type: int {

        do update_beliefs;

        compliance_probability <- base_compliance;

        if (has_belief(food_pressure)) {
            compliance_probability <- compliance_probability - food_pressure_penalty;
        }

        if (has_belief(tree_benefit)) {
            compliance_probability <- compliance_probability + tree_knowledge_bonus;
        }

        compliance_probability <- max(0.0, min(1.0, compliance_probability));

        if (a_recommended = fallback_action) {
            complied <- true;
            return a_recommended;
        }

        if (rnd(1.0) <= compliance_probability) {
            complied <- true;
            return a_recommended;
        } else {
            complied <- false;
            return fallback_action;
        }
    }
}

/*
 * The Bandit represents the risk of tree cutting (p_cut).
 * Currently a simple stochastic agent (flip(p_cut)). Can later 
 * evolve into a strategic agent with its own decision logic.
 */
species Bandit parent: Actor {
	float cut_probability <- 0.0;
}

/*
 * The Resource concept represents the resources that actors use. It is
 * associated with the concept Actor through the ownedResources association.
 */
species Resource {
    string name;
}

/*
 * The NaturalResource concept represents natural elements
 * that actors use. Water, Tree, and Land inherit from this concept.
 */
species NaturalResource parent: Resource {
    string type;
    float quantity;
}

/*
 * The Land concept represents areas of soil used by actors for grazing,
 * cultivation, or settlement.
 */
species Land parent: NaturalResource {
	string type -> "Land";
}

/*
 * The Tree concept represents trees on the landscape, 
 * either naturally occurring or planted through RNA.
 */
species Tree parent: NaturalResource {
	string type -> "Tree";
	
	float size <- 1.0;
    rgb color <- #green;
	
	aspect base {
		draw circle(size) color: color;
    }
}

/*
 * The Parcel concept represents a spatial unit of land that can be used
 * for agricultural, pastoral, or other land-based activities.
 * Grid dimensions are parameterized via global variables H and W.
 */
grid Parcel parent: Land width: W height: H {
	
	// Context variable (static per episode)
    int soil_type_id;       // Context c: soil type determining reward profile
    
    // State variable (dynamic)
    int new_tree_age;       // State s: tree age (0 = no tree, up to nS-1)
    
    // Biophysical variables (for future composite reward)
    float C_input;    // Carbon input flux (tC/ha/yr)
    float yield;            // Crop yield (t/ha)
}
