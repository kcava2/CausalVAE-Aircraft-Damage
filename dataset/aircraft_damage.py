"""
Aircraft Damage Dataset — CausalVAE (simplified, 4 supervised concepts)

4 concepts, all directly supervised by YOLO class labels:
    0: crack
    1: dent
    2: paint_off
    3: scratch

Severity target: total damage instance count from YOLO boxes
(e.g. 3 cracks = 3, 1 crack + 1 dent = 2)

YOLO class remapping (missing_head excluded):
    YOLO 0 → concept 0  (crack)
    YOLO 1 → concept 1  (dent)
    YOLO 2 → skip       (missing_head)
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
N_CONCEPTS   = 4
N_OBSERVABLE = 4   # all concepts are supervised

CLASS_NAMES = [
    'crack',     # 0
    'dent',      # 1
    'paint_off', # 2
    'scratch',   # 3
]

# scale[j] = [mean, half_range] used by condition_prior():
#   normalised = (label - mean) / half_range
SCALE = np.array([
    [0.5, 0.5],  # crack         0/1 → [-1, +1]
    [0.5, 0.5],  # dent          0/1 → [-1, +1]
    [0.5, 0.5],  # paint_off     0/1 → [-1, +1]
    [0.5, 0.5],  # scratch       0/1 → [-1, +1]
], dtype=float)

# YOLO class_id → concept index (None = skip)
_YOLO_TO_CONCEPT = {0: 0, 1: 1, 2: None, 3: 2, 4: 3}

IMAGE_SIZE = 64


# ── Label parsing ─────────────────────────────────────────────────────────────

def parse_yolo_label(label_path: str) -> np.ndarray:
    """
    Read a YOLO .txt label file and return a binary array of length N_CONCEPTS.
    Each line: class_id  x  y  w  h  → sets concept flag to 1 if present.
    Returns float32 array shape (N_CONCEPTS,).
    """
    obs = np.zeros(N_CONCEPTS, dtype=np.float32)
    if not os.path.exists(label_path):
        return obs
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id = int(line.split()[0])
            concept  = _YOLO_TO_CONCEPT.get(class_id)
            if concept is not None:
                obs[concept] = 1.0
    return obs


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
        label:       FloatTensor (4,)           binary concept flags
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

        obs       = parse_yolo_label(lbl_path)          # (4,) binary
        sev_count = count_damage_instances(lbl_path)    # int

        label_tensor = torch.tensor(obs, dtype=torch.float32)
        sev_tensor   = torch.tensor(float(sev_count), dtype=torch.float32)

        return img_tensor, label_tensor, sev_tensor

    def class_distribution(self) -> dict:
        counts = np.zeros(N_OBSERVABLE, dtype=float)
        total  = len(self.samples)
        for _, lbl_path in self.samples:
            counts += parse_yolo_label(lbl_path)
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
