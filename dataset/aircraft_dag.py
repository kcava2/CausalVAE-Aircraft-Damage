"""
Aircraft Damage Causal DAG — 4 supervised concepts.

Concept indices:
    0: crack
    1: dent
    2: paint_off
    3: scratch

Causal edges (A[i][j] = 1 means concept i causally influences j):
    dent  (1) → scratch   (3)  — impact that dents a surface often scratches it
    crack (0) → paint_off (2)  — cracks expose metal, accelerating paint loss
    dent  (1) → paint_off (2)  — dent stress can lift or crack paint
"""

import torch

N_CONCEPTS = 4


def get_dag_init() -> torch.Tensor:
    """Return the 4×4 adjacency matrix for the aircraft damage causal graph."""
    A = torch.zeros(N_CONCEPTS, N_CONCEPTS)
    A[1, 3] = 1.0   # dent  → scratch
    A[0, 2] = 1.0   # crack → paint_off
    A[1, 2] = 1.0   # dent  → paint_off
    return A


# Pre-built constant for import convenience
A_INIT: torch.Tensor = get_dag_init()
