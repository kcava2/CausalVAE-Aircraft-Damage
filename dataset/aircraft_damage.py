"""
Aircraft Damage Dataset — CausalVAE (4 observable + 3 latent root-cause concepts)

Observable concepts (directly supervised by YOLO class labels):
    0: crack
    1: dent
    2: paint_off
    3: scratch

Latent root-cause concepts (weakly supervised via physics-informed co-occurrence):
    4: impact_force   — derived from dent/scratch/crack co-occurrence
    5: metal_fatigue  — derived from crack-without-dent pattern
    6: corrosion      — derived from paint_off-without-impact pattern

Severity target: total damage instance count from YOLO boxes
(e.g. 3 cracks = 3, 1 crack + 1 dent = 2)

YOLO class remapping (missing_head excluded from observable labels):
    YOLO 0 → concept 0  (crack)
    YOLO 1 → concept 1  (dent)
    YOLO 2 → skip       (missing_head — used only in metal_fatigue derivation)
    YOLO 3 → concept 2  (paint_off)
    YOLO 4 → concept 3  (scratch)
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as Data
from PIL import Image
from torchvision import transforms

# ── Constants ─────────────────────────────────────────────────────────────────
N_CONCEPTS   = 7
N_OBSERVABLE = 4   # only first 4 concepts have direct YOLO supervision

CLASS_NAMES = [
    'crack',        # 0 — observable
    'dent',         # 1 — observable
    'paint_off',    # 2 — observable
    'scratch',      # 3 — observable
    'impact_force', # 4 — latent root cause
    'metal_fatigue',# 5 — latent root cause
    'corrosion',    # 6 — latent root cause
]

# scale[j] = [mean, half_range] used by condition_prior():
#   normalised = (label - mean) / half_range
# Observable (0-3): binary 0/1 → normalised [-1, +1]
# Latent   (4-6): soft [0,1] → prior mean tracks label value
SCALE = np.array([
    [0.5, 0.5],  # crack         0/1 → [-1, +1]
    [0.5, 0.5],  # dent          0/1 → [-1, +1]
    [0.5, 0.5],  # paint_off     0/1 → [-1, +1]
    [0.5, 0.5],  # scratch       0/1 → [-1, +1]
    [0.0, 1.0],  # impact_force  [0,1] → prior mean = label value
    [0.0, 1.0],  # metal_fatigue [0,1] → prior mean = label value
    [0.0, 1.0],  # corrosion     [0,1] → prior mean = label value
], dtype=float)

# YOLO class_id → concept index (None = skip)
_YOLO_TO_CONCEPT = {0: 0, 1: 1, 2: None, 3: 2, 4: 3}

IMAGE_SIZE = 64


# ── Weak supervision ──────────────────────────────────────────────────────────

def infer_latent_labels(presence_5dim: np.ndarray) -> np.ndarray:
    """
    Derive soft labels for the 3 latent root-cause concepts from the raw
    5-dim YOLO class presence vector (indexed by YOLO class ID):
        presence_5dim[0] = crack
        presence_5dim[1] = dent
        presence_5dim[2] = missing_head
        presence_5dim[3] = paint_off
        presence_5dim[4] = scratch

    Physics-informed rules:
        impact_force:  dents are definitionally caused by external force;
                       co-occurring crack+dent is a strong impact indicator.
        metal_fatigue: cracks without dents are the classic fatigue-failure pattern.
        corrosion:     paint loss without mechanical damage indicates chemical
                       degradation rather than impact.

    Returns float32 array of shape (3,) with values in [0, 1].
    """
    crack        = float(presence_5dim[0])
    dent         = float(presence_5dim[1])
    missing_head = float(presence_5dim[2])
    paint_off    = float(presence_5dim[3])
    scratch      = float(presence_5dim[4])

    impact_force   = min(dent * 1.0 + scratch * 0.5 + crack * dent * 0.5, 1.0)
    metal_fatigue  = min(crack * (1.0 - dent) + missing_head * 1.0, 1.0)
    corrosion      = min(paint_off * 1.0 + paint_off * (1.0 - dent) * 0.5, 1.0)

    return np.array([impact_force, metal_fatigue, corrosion], dtype=np.float32)


# ── Label parsing ─────────────────────────────────────────────────────────────

def parse_yolo_label(label_path: str) -> np.ndarray:
    """
    Read a YOLO .txt label file and return a 7-dim concept vector:
        [crack, dent, paint_off, scratch,  ← 4 observable (binary)
         impact_force, metal_fatigue, corrosion]  ← 3 latent (soft [0,1])

    Each line in the file: class_id  x  y  w  h
    Returns float32 array shape (N_CONCEPTS,).
    """
    # 5-dim raw presence indexed by YOLO class id (0=crack,1=dent,2=missing_head,3=paint_off,4=scratch)
    presence = np.zeros(5, dtype=np.float32)

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                class_id = int(line.split()[0])
                if 0 <= class_id <= 4:
                    presence[class_id] = 1.0

    # Observable 4-dim (YOLO 0→crack, 1→dent, 3→paint_off, 4→scratch)
    obs = np.array([presence[0], presence[1], presence[3], presence[4]], dtype=np.float32)

    # Latent 3-dim (weakly supervised)
    latent = infer_latent_labels(presence)

    return np.concatenate([obs, latent])  # shape (7,)


def count_damage_instances(label_path: str) -> int:
    """
    Count total individual damage detections in the YOLO label file,
    excluding missing_head (YOLO class 2).

    Each line in the file is one bounding box, so 3 cracks = 3 lines = count 3.
    Returns an integer >= 0.
    """
    if not os.path.exists(label_path):
        return 0
    count = 0
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id = int(line.split()[0])
            if _YOLO_TO_CONCEPT.get(class_id) is not None:
                count += 1
    return count


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(split: str):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class AircraftDamageDataset(Data.Dataset):
    """
    Returns per sample:
        img_tensor:  FloatTensor (3, 64, 64)   normalised RGB
        label:       FloatTensor (7,)           4 binary observable + 3 soft latent concepts
        sev_count:   FloatTensor scalar         total damage instance count
    """
    def __init__(self, root: str, split: str = 'train'):
        self.root      = root
        self.split     = split
        self.transform = get_transforms(split)

        img_dir = Path(root) / split / 'images'
        lbl_dir = Path(root) / split / 'labels'

        exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        img_files = sorted(p for p in img_dir.iterdir() if p.suffix in exts)

        self.samples = []
        for img_path in img_files:
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            self.samples.append((str(img_path), str(lbl_path)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        img        = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)

        concept_vec = parse_yolo_label(lbl_path)          # (7,) float32
        sev_count   = count_damage_instances(lbl_path)    # int

        label_tensor = torch.tensor(concept_vec, dtype=torch.float32)
        sev_tensor   = torch.tensor(float(sev_count), dtype=torch.float32)

        return img_tensor, label_tensor, sev_tensor

    def class_distribution(self) -> dict:
        counts = np.zeros(N_OBSERVABLE, dtype=float)
        total  = len(self.samples)
        for _, lbl_path in self.samples:
            counts += parse_yolo_label(lbl_path)[:N_OBSERVABLE]
        rates = counts / max(total, 1)
        return {CLASS_NAMES[i]: float(rates[i]) for i in range(N_OBSERVABLE)}


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloader(root: str, split: str, batch_size: int,
                   num_workers: int = 4) -> Data.DataLoader:
    ds      = AircraftDamageDataset(root, split)
    shuffle = (split == 'train')
    return Data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=(split == 'train'),
        pin_memory=torch.cuda.is_available(),
    )
