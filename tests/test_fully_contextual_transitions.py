"""
Test fully-contextual transition kernels.

Usage:
    docker-compose exec gym-agent python tests/test_fully_contextual_transitions.py
"""

import os
import sys
import asyncio
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)


def log_test(name, status, detail=None):
    line = f"[TEST] {name:<55} {status}"
    print(line)
    if detail:
        print(f"       {detail}")


async def main():
    gaml_path = os.environ.get(
        "GAML_PATH",
        "/usr/lib/gama/workspace/gama_models/EcoSysML/models/main.gaml"
    )

    nS = 4
    nA = 4
    nC = 3
    trigger_action = 2
    p_cut = 0.15
    context_p_cut_scale_gap = 0.05

    print("=" * 70)
    print("Fully-Contextual Transition Test")
    print(f"  GAML:      {gaml_path}")
    print(f"  nS={nS}, nA={nA}, nC={nC}")
    print(f"  p_cut:     {p_cut}")
    print(f"  gap:       {context_p_cut_scale_gap}")
    print("=" * 70)

    try:
        from src.contextual_stat_rl.environments.gama_register import make_gama

        env = make_gama(
            "gama-agrocarbon-fully-contextual",
            nS=nS,
            nA=nA,
            nC=nC,
            trigger_action=trigger_action,
            p_cut=p_cut,
            difficulty="easy",
            context_p_cut_scale_gap=context_p_cut_scale_gap,
            reference_context=0,
            c_is_static=True,
            gaml_experiment_path=gaml_path,
        )

        log_test("1. Environment creation", "OK")
    except Exception as e:
        log_test("1. Environment creation", "FAILED", str(e))
        sys.exit(1)

    try:
        assert env.p_is_contextual is True, "Expected p_is_contextual=True"
        assert env.r_is_contextual is True, "Expected r_is_contextual=True"
        log_test("2. Fully-contextual flags", "OK",
                 f"p_is_contextual={env.p_is_contextual}, r_is_contextual={env.r_is_contextual}")
    except Exception as e:
        log_test("2. Fully-contextual flags", "FAILED", str(e))
        env.close()
        sys.exit(1)

    try:
        s = 1
        a = 0

        transitions = []
        for c in range(nC):
            P_c = env.getTransition(s, a, c=c)
            transitions.append(P_c)

            assert isinstance(P_c, np.ndarray), f"Expected ndarray, got {type(P_c)}"
            assert len(P_c) == nS, f"Transition length should be {nS}, got {len(P_c)}"
            assert abs(P_c.sum() - 1.0) < 1e-8, f"Transition does not sum to 1: {P_c}"

        log_test("3. Contextual transitions are valid", "OK",
                 f"P(c=0)={transitions[0]}, P(c=1)={transitions[1]}, P(c=2)={transitions[2]}")
    except Exception as e:
        log_test("3. Contextual transitions are valid", "FAILED", str(e))
        env.close()
        sys.exit(1)

    try:
        # For s=1, a=0, natural next state is 2.
        # Cutting risk sends the state back to 0.
        p0 = transitions[0][0]
        p1 = transitions[1][0]
        p2 = transitions[2][0]

        expected = [
            p_cut * (1.0 + context_p_cut_scale_gap * c)
            for c in range(nC)
        ]

        assert np.allclose([p0, p1, p2], expected, atol=1e-8), (
            f"Expected contextual p_cut values {expected}, "
            f"got {[p0, p1, p2]}"
        )

        assert p0 < p1 < p2, (
            f"Expected increasing cutting risk across contexts, got {[p0, p1, p2]}"
        )

        log_test("4. Cutting risk varies by context", "OK",
                 f"observed={[round(x, 4) for x in [p0, p1, p2]]}, "
                 f"expected={[round(x, 4) for x in expected]}")
    except Exception as e:
        log_test("4. Cutting risk varies by context", "FAILED", str(e))
        env.close()
        sys.exit(1)

    try:
        env.close()
        log_test("5. Clean close", "OK")
    except Exception as e:
        log_test("5. Clean close", "FAILED", str(e))
        sys.exit(1)

    print()
    print("=" * 70)
    print("[TEST] Fully-contextual transition tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())