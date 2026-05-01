"""
ContextualGamaEnv: A contextual MDP environment that delegates simulation to GAMA.

This class inherits from ContextualDiscreteMDP to maintain full compatibility
with existing RL agents and the oracle (which reads P and R locally), but
overrides step() and reset() to use GAMA headless as the simulation engine.

Designed for the sequential case. The observation format remains
(c, s) as expected by all contextual agents.
"""

import numpy as np
import gama_client
from gama_gymnasium.gama_client_wrapper import GamaClientWrapper
from src.contextual_stat_rl.environments.ContextualMDPs_discrete.contextualMDP import ContextualDiscreteMDP
from statisticalrl_environments.MDPs_discrete.utils import categorical_sample


class ContextualGamaEnv(ContextualDiscreteMDP):
    """
    Contextual MDP environment backed by a GAMA headless simulation.
 
    Inherits ContextualDiscreteMDP so that:
    - The oracle can call getTransition(s, a) and getMeanReward(s, a) on local P/R.
    - RL agents interact via the standard step()/reset() interface.
    - The experiment runner does not need to change.
 
    The step() and reset() methods are overridden to communicate with GAMA
    instead of using local P and R for transitions and rewards.
 
    Parameters
    ----------
    gaml_experiment_path : str
        Absolute path to the .gaml model file on the GAMA server.
    gaml_experiment_name : str
        Name of the experiment to run (e.g. "gym_env").
    gama_ip_address : str
        IP address of the GAMA headless server.
    gama_port : int
        Port of the GAMA headless server.
    gaml_experiment_parameters : list of dict, optional
        Additional parameters to pass to the GAMA experiment at load time.
    **kwargs
        All standard ContextualDiscreteMDP arguments (nS, nA, nC, P, R, mu0,
        nu, skeleton, c_is_static, p_is_contextual, r_is_contextual,
        nameActions, seed, name).
    """
 
    def __init__(self, gaml_experiment_path, gaml_experiment_name, 
             gama_ip_address="localhost", gama_port=6868,
             gaml_experiment_parameters=None, **kwargs):
        # 1. Safety default
        self.gama_client = None
        
        # 2. Connect to GAMA BEFORE parent init
        self.gaml_file_path = gaml_experiment_path
        self.experiment_name = gaml_experiment_name
        self.gama_ip_address = gama_ip_address
        self.gama_port = gama_port
        self.gaml_experiment_parameters = gaml_experiment_parameters or []
        
        self.gama_client = GamaClientWrapper(gama_ip_address, gama_port)
        self.experiment_id = self.gama_client.load_experiment(
            self.gaml_file_path,
            self.experiment_name,
            self.gaml_experiment_parameters,
        )
        
        # 3. Parent init
        super().__init__(**kwargs)

 
    def _restart_gama_experiment(self, seed=None):
        """
        Stop current GAMA experiment and load a fresh one with a seed parameter.
        Slower than reload, but safer for RNG reproducibility.
        """
        from gama_client.message_types import MessageTypes

        # 1. Stop current experiment
        if self.experiment_id is not None:
            response = self.gama_client.client.stop(self.experiment_id)
            if response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
                raise RuntimeError(f"GAMA stop failed: {response}")

        # 2. Rebuild parameters
        parameters = list(self.gaml_experiment_parameters)

        if seed is not None:
            parameters.append({
                "type": "float",
                "name": "Forced Seed",
                "value": str(float(seed)),
            })

        # 3. Load fresh experiment
        self.experiment_id = self.gama_client.load_experiment(
            self.gaml_file_path,
            self.experiment_name,
            parameters,
        )
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.
 
        Protocol:
        1. Sample context c and initial state s locally (Python controls the seed).
        2. Reload the GAMA experiment.
        3. Write pending_contexts, pending_states, reset_requested via expressions.
        4. Trigger one GAMA step so the apply_python_reset reflex executes.
        5. Read gym_interface.data and assert coherence.
        6. Return ((c, s), info).
        """
        # Handle seeding
        if seed is not None:
            self.np_random, _ = np.random.default_rng(seed), seed
 
        # Sample context and initial state locally
        self.c = categorical_sample(self.nu, self.np_random)
        self.s = categorical_sample(self.mu0[self.c], self.np_random)
        self.lastaction = None
 
        # Reload GAMA experiment with seed passed at initialization
        self._restart_gama_experiment(seed)

        # Write reset variables
        self.gama_client._execute_expression(
            self.experiment_id,
            f"pending_contexts <- [{self.c}];"
        )
        self.gama_client._execute_expression(
            self.experiment_id,
            f"pending_states <- [{self.s}];"
        )
        self.gama_client._execute_expression(
            self.experiment_id,
            "reset_requested <- true;"
        )
 
        # Trigger one GAMA step to execute the apply_python_reset reflex
        from gama_client.message_types import MessageTypes
        response = self.gama_client.client.step(self.experiment_id, sync=True)
        if response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise RuntimeError(f"GAMA reset step failed: {response}")
 
        # Read the updated gym_interface.data
        reset_data = self.gama_client._execute_expression(
            self.experiment_id,
            r"gym_interface.data"
        )
 
        # Verify coherence between Python and GAMA
        gama_state = reset_data["State"]
        gama_c = int(gama_state[0][0])
        gama_s = int(gama_state[0][1])
        assert gama_c == self.c, (
            f"Context mismatch: Python={self.c}, GAMA={gama_c}"
        )
        assert gama_s == self.s, (
            f"State mismatch: Python={self.s}, GAMA={gama_s}"
        )
 
        return (self.c, self.s), {"mean": 0.0}
 
    def step(self, action):
        """
        Execute one step via GAMA.
 
        1. Send [action] to GAMA (list format for batch compatibility).
        2. GAMA executes StepTask + CutTask on the parcel.
        3. Parse the response to extract (c, s), reward, info.
        4. Update internal state.
 
        Parameters
        ----------
        action : int
            The action to execute.
 
        Returns
        -------
        observation : tuple (int, int)
            The contextual observation (c, s).
        reward : float
            The reward sampled by GAMA.
        done : bool
            Whether the episode is terminated.
        truncated : bool
            Whether the episode is truncated.
        info : dict
            Additional information including action_executed, was_cut, mean reward.
        """
        # Send action as a list (batch-ready format)
        gama_action = [int(action)]
 
        # Execute step in GAMA
        step_data = self.gama_client.execute_step(self.experiment_id, gama_action)
 
        # Compute true mean reward from local P/R BEFORE updating state
        # (reward depends on state at time of action, not next state)
        mean_reward = self.getMeanReward(self.s, action)
 
        # Parse GAMA response
        observation, reward, terminated, truncated, info = self._parse_step_response(
            step_data
        )
 
        # Inject true mean reward from local model
        info["mean"] = mean_reward
 
        # Update internal state
        self.s = observation[1]  # tree_age
        self.c = observation[0]  # context (static, but read for consistency)
        self.lastaction = action
        self.lastreward = reward
 
        # Resample context if dynamic
        if not self.c_is_static:
            self.c = categorical_sample(self.nu, self.np_random)
 
        return observation, reward, terminated, truncated, info

 
    def _parse_step_response(self, step_data):
        """
        Parse the GAMA step response into the contextual MDP format.

        GAMA returns:
            State: [[c, s]]
            Reward: float
            Terminated: bool
            Truncated: bool
            Info: map with parcel_info, actions_recommended, actions_executed, etc.

        For the sequential case (1 parcel), we extract the first element.
        """
        # Extract state: [[c, s]] -> (c, s)
        raw_state = step_data["State"]
        if isinstance(raw_state, list) and len(raw_state) > 0:
            parcel_obs = raw_state[0]
            c = int(parcel_obs[0])
            s = int(parcel_obs[1])
        else:
            raise ValueError(f"Unexpected state format from GAMA: {raw_state}")

        observation = (c, s)
        reward = float(step_data["Reward"])
        terminated = bool(step_data["Terminated"])
        truncated = bool(step_data["Truncated"])

        raw_info = step_data.get("Info", {})
        info = {}

        if isinstance(raw_info, dict):
            # Top-level action lists
            if "actions_recommended" in raw_info:
                info["action_recommended"] = int(raw_info["actions_recommended"][0])

            if "actions_executed" in raw_info:
                info["action_executed"] = int(raw_info["actions_executed"][0])

            if "cut_flags" in raw_info:
                info["was_cut"] = bool(raw_info["cut_flags"][0])

            if "parcel_rewards" in raw_info:
                info["parcel_reward"] = float(raw_info["parcel_rewards"][0])

            # Farmer-level compliance info
            if "compliance_probability" in raw_info:
                info["compliance_probability"] = float(raw_info["compliance_probability"])

            if "farmer_complied" in raw_info:
                info["farmer_complied"] = bool(raw_info["farmer_complied"])

            if "household_size" in raw_info:
                info["household_size"] = int(raw_info["household_size"])

            if "tree_knowledge" in raw_info:
                info["tree_knowledge"] = float(raw_info["tree_knowledge"])

            # Parcel-level info, if available
            if "parcel_info" in raw_info and len(raw_info["parcel_info"]) > 0:
                pinfo = raw_info["parcel_info"][0]

                if isinstance(pinfo, dict):
                    info["parcel_info"] = pinfo

                    if "action_recommended" in pinfo:
                        info["action_recommended"] = int(pinfo["action_recommended"])

                    if "action_executed" in pinfo:
                        info["action_executed"] = int(pinfo["action_executed"])

                    if "complied" in pinfo:
                        info["complied"] = bool(pinfo["complied"])

                    if "reward" in pinfo:
                        info["parcel_reward"] = float(pinfo["reward"])

                    if "was_cut" in pinfo:
                        info["was_cut"] = bool(pinfo["was_cut"])

        return observation, reward, terminated, truncated, info
 
    def close(self):
        """Close the GAMA connection and clean up resources."""
        if hasattr(self, "gama_client") and self.gama_client is not None:
            try:
                self.gama_client.close()
            except Exception as e:
                print(f"Warning: Error closing GAMA connection: {e}")
            finally:
                self.gama_client = None