"""
Handshake test: verifies connectivity between the Python container 
and the GAMA headless server container.

Usage (from inside the gym-agent container):
    python test_handshake.py

Or from host:
    docker-compose exec gym-agent python test_handshake.py

Expected output:
    [HANDSHAKE] Step 1/5: Connecting to GAMA server...          OK
    [HANDSHAKE] Step 2/5: Loading experiment...                  OK
    [HANDSHAKE] Step 3/5: Reading initial state...               OK
    [HANDSHAKE] Step 4/5: Triggering reset_from_python...        OK
    [HANDSHAKE] Step 5/5: Executing one step...                  OK
    [HANDSHAKE] All checks passed!
"""

import os
import sys
import time
import asyncio


def log(step, total, message, status=None):
    """Print a formatted log line."""
    prefix = f"[HANDSHAKE] Step {step}/{total}: {message}"
    if status:
        print(f"{prefix:<55} {status}")
    else:
        print(prefix)


async def main():
    # -------------------------------------------------------
    # Configuration (from env vars or defaults)
    # -------------------------------------------------------
    gama_host = os.environ.get("GAMA_HOST", "localhost")
    gama_port = int(os.environ.get("GAMA_PORT", 6868))
 
    # Path to main.gaml as seen by the GAMA container
    gaml_path = os.environ.get(
        "GAML_PATH",
        "/usr/lib/gama/workspace/gama_models/EcoSysML/models/main.gaml"
    )
    experiment_name = "gym_env"
 
    total_steps = 5
 
    print("=" * 60)
    print("GAMA Headless Handshake Test")
    print(f"  Host: {gama_host}")
    print(f"  Port: {gama_port}")
    print(f"  GAML: {gaml_path}")
    print(f"  Experiment: {experiment_name}")
    print("=" * 60)
 
    # -------------------------------------------------------
    # Step 1: Connect to GAMA server
    # -------------------------------------------------------
    try:
        from gama_gymnasium.gama_client_wrapper import GamaClientWrapper
 
        client = GamaClientWrapper(gama_host, gama_port)
        log(1, total_steps, "Connecting to GAMA server...", "OK")
    except Exception as e:
        log(1, total_steps, "Connecting to GAMA server...", "FAILED")
        print(f"  Error: {e}")
        print("\n  Check that:")
        print(f"    - GAMA headless is running on {gama_host}:{gama_port}")
        print("    - docker-compose is up: docker-compose up -d")
        print("    - the gym-agent container can reach gama-headless")
        sys.exit(1)
 
    # -------------------------------------------------------
    # Step 2: Load the experiment
    # -------------------------------------------------------
    try:
        experiment_id = client.load_experiment(
            gaml_path, experiment_name
        )
        log(2, total_steps, "Loading experiment...", "OK")
        print(f"  Experiment ID: {experiment_id}")
    except Exception as e:
        log(2, total_steps, "Loading experiment...", "FAILED")
        print(f"  Error: {e}")
        print("\n  Check that:")
        print(f"    - {gaml_path} exists inside the GAMA container")
        print(f"    - experiment '{experiment_name}' is defined in main.gaml")
        print("    - the gama_models volume is correctly mounted")
        client.close()
        sys.exit(1)
 
    # -------------------------------------------------------
    # Step 3: Read initial state
    # -------------------------------------------------------
    try:
        state = client.get_state(experiment_id)
        info = client.get_info(experiment_id)
        log(3, total_steps, "Reading initial state...", "OK")
        print(f"  State: {state}")
        print(f"  Info:  {info}")
    except Exception as e:
        log(3, total_steps, "Reading initial state...", "FAILED")
        print(f"  Error: {e}")
        client.close()
        sys.exit(1)
 
    # -------------------------------------------------------
    # Step 4: Reset via variable assignments + step trigger
    # -------------------------------------------------------
    try:
        c_test, s_test = 0, 0
 
        # Write reset variables
        client._execute_expression(experiment_id, f"pending_contexts <- [{c_test}];")
        client._execute_expression(experiment_id, f"pending_states <- [{s_test}];")
        client._execute_expression(experiment_id, "reset_requested <- true;")
 
        # Trigger one GAMA step to execute the apply_python_reset reflex
        from gama_client.message_types import MessageTypes
        response = client.client.step(experiment_id, sync=True)
        if response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise RuntimeError(f"GAMA reset step failed: {response}")
 
        # Read back gym_interface.data
        reset_data = client._execute_expression(experiment_id, r"gym_interface.data")
 
        log(4, total_steps, "Resetting via apply_python_reset...", "OK")
        print(f"  Context sent: {c_test}")
        print(f"  State sent:   {s_test}")
        print(f"  Reset data:   {reset_data}")
 
        # Verify coherence
        gama_c = int(reset_data["State"][0][0])
        gama_s = int(reset_data["State"][0][1])
        assert gama_c == c_test, f"Context mismatch: sent={c_test}, got={gama_c}"
        assert gama_s == s_test, f"State mismatch: sent={s_test}, got={gama_s}"
        print(f"  Coherence check: PASSED (c={gama_c}, s={gama_s})")
 
    except Exception as e:
        log(4, total_steps, "Resetting via apply_python_reset...", "FAILED")
        print(f"  Error: {e}")
        print("\n  Check that:")
        print("    - pending_contexts, pending_states, reset_requested are in global")
        print("    - reflex apply_python_reset is defined in main.gaml")
        client.close()
        sys.exit(1)
 
    # -------------------------------------------------------
    # Step 5: Execute one step with a default action
    # -------------------------------------------------------
    try:
        test_action = [2]  # List format, one action for one parcel
        step_data = client.execute_step(experiment_id, test_action)
        log(5, total_steps, "Executing one step...", "OK")
        print(f"  Action sent: {test_action}")
        print(f"  State:       {step_data.get('State')}")
        print(f"  Reward:      {step_data.get('Reward')}")
        print(f"  Terminated:  {step_data.get('Terminated')}")
        print(f"  Info:        {step_data.get('Info')}")
    except Exception as e:
        log(4, total_steps, "Executing one step...", "FAILED")
        print(f"  Error: {e}")
        client.close()
        sys.exit(1)
 
    # -------------------------------------------------------
    # Done
    # -------------------------------------------------------
    client.close()
    print()
    print("=" * 60)
    print("[HANDSHAKE] All checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    # Boots up the event loop required by nest_asyncio in the GAMA client
    asyncio.run(main())