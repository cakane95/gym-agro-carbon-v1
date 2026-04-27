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
 * The Farmer is the owner/operator of a parcel. Currently passive 
 * (executes the RL recommendation directly). Will become a BDI agent 
 * that can override the recommended action based on household constraints.
 */
species Farmer parent: Actor {
	Parcel my_parcel;
	int household_size <- 1;
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
