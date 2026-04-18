"""
Aircraft Damage Causal DAG — 4 observable + 3 latent root-cause concepts.

Concept indices:
    0: crack          (observable)
    1: dent           (observable)
    2: paint_off      (observable)
    3: scratch        (observable)
    4: impact_force   (latent root cause)
    5: metal_fatigue  (latent root cause)
    6: corrosion      (latent root cause)

Causal edges (A[i][j] = 1 means concept i causally influences j):
    impact_force  (4) → crack      (0)  — impact stress can initiate cracking
    impact_force  (4) → dent       (1)  — dents are definitionally caused by impact
    impact_force  (4) → scratch    (3)  — impact sliding creates scratches
    metal_fatigue (5) → crack      (0)  — fatigue failure presents as cracking
    corrosion     (6) → paint_off  (2)  — corrosion lifts and removes paint
    dent          (1) → crack      (0)  — dent stress concentrations initiate cracks
    dent          (1) → scratch    (3)  — impact that dents a surface often scratches it
    scratch       (3) → paint_off  (2)  — scratches expose metal, accelerating paint loss
"""

import torch

N_CONCEPTS = 7

CLASS_NAMES = [
    'crack',         # 0
    'dent',          # 1
    'paint_off',     # 2
    'scratch',       # 3
    'impact_force',  # 4
    'metal_fatigue', # 5
    'corrosion',     # 6
]


def get_dag_init() -> torch.Tensor:
    """Return the 7×7 adjacency matrix for the aircraft damage causal graph."""
    A = torch.zeros(N_CONCEPTS, N_CONCEPTS)
    # Latent root causes → observable effects
    A[4, 0] = 1.0   # impact_force  → crack
    A[4, 1] = 1.0   # impact_force  → dent
    A[4, 3] = 1.0   # impact_force  → scratch
    A[5, 0] = 1.0   # metal_fatigue → crack
    A[6, 2] = 1.0   # corrosion     → paint_off
    # Observable → observable
    A[1, 0] = 1.0   # dent          → crack
    A[1, 3] = 1.0   # dent          → scratch
    A[3, 2] = 1.0   # scratch       → paint_off
    return A


# Pre-built constant for import convenience
A_INIT: torch.Tensor = get_dag_init()
