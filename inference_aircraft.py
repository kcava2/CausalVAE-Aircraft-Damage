"""
Aircraft Damage CausalVAE — Inference & Counterfactual Analysis

Usage:
    python inference_aircraft.py --image path/to/img.jpg --checkpoint checkpoints/aircraft_best.pt
    python inference_aircraft.py --image_dir path/to/dir/ --checkpoint checkpoints/aircraft_best.pt

Outputs per image (in --out_dir, default ./inference_results/):
    <stem>_analysis.png        — original | reconstruction + probability bars + root causes
    <stem>_counterfactuals.png — original + 4 counterfactual columns with delta bars
    <stem>_report.json         — full structured report
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from codebase.models.mask_vae_aircraft import CausalVAE
from codebase import utils as ut
from models.classifier_head import MultiLabelHead, SeverityHead
from dataset.aircraft_damage import IMAGE_SIZE, N_OBSERVABLE, CLASS_NAMES, SCALE

# ── Constants ─────────────────────────────────────────────────────────────────
OBS_NAMES    = CLASS_NAMES[:N_OBSERVABLE]
LATENT_NAMES = CLASS_NAMES[N_OBSERVABLE:]  # impact_force, metal_fatigue, corrosion

OBS_COLORS    = ['#DC3232', '#FFA500', '#32B432', '#508CFF']
LATENT_COLORS = ['#FF6B6B', '#4ECDC4', '#95E1D3']
BG            = '#1c1c1c'

_MAINTENANCE = {
    'impact_force':  'Inspect for structural deformation. Check adjacent panels.',
    'metal_fatigue': 'Systemic issue. Review load cycle history. Inspect adjacent structure.',
    'corrosion':     'Environmental damage. Treat surface. Inspect surrounding panels.',
}


def severity_label(count: int) -> str:
    if count == 0:
        return 'None'
    elif count <= 2:
        return f'Minor — {count} damage instance(s)'
    elif count <= 4:
        return f'Moderate — {count} damage instances'
    else:
        return f'Severe — {count} damage instances'


# Causal downstream map (from DAG)
_CAUSAL_DOWNSTREAM = {
    'crack':        [],
    'dent':         ['crack', 'scratch'],
    'paint_off':    [],
    'scratch':      ['paint_off'],
    'impact_force': ['crack', 'dent', 'scratch'],
    'metal_fatigue':['crack'],
    'corrosion':    ['paint_off'],
}

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image',      type=str,   default=None)
parser.add_argument('--image_dir',  type=str,   default=None)
parser.add_argument('--checkpoint', type=str,   default='checkpoints/aircraft_best.pt')
parser.add_argument('--out_dir',    type=str,   default='./inference_results')
parser.add_argument('--threshold',       type=float, default=0.5)
parser.add_argument('--recompute_corr',  action='store_true',
                    help='Force recomputation of corr_idx.pt even if it already exists')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ── Load model ────────────────────────────────────────────────────────────────
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
cfg  = ckpt.get('config', {'z_dim': 28, 'z1_dim': 7, 'z2_dim': 4})
Z_DIM, Z1_DIM, Z2_DIM = cfg['z_dim'], cfg['z1_dim'], cfg['z2_dim']
_N_OBS = cfg.get('n_observable', N_OBSERVABLE)

lvae = CausalVAE(
    name='aircraft_causalvae',
    z_dim=Z_DIM, z1_dim=Z1_DIM, z2_dim=Z2_DIM,
    inference=True,
    scale=SCALE,
    initial=False,
).to(device)
missing, unexpected = lvae.load_state_dict(ckpt['lvae'], strict=False)
if missing:
    print(f'  [WARN] Missing keys in checkpoint (will use random init): {missing[:4]}...')
if unexpected:
    print(f'  [WARN] Unexpected keys ignored: {unexpected[:4]}...')
lvae.eval()

# ── Load skip-channel correlations ───────────────────────────────────────────
_corr_path = Path(args.checkpoint).parent / 'corr_idx.pt'
if _corr_path.exists() and not args.recompute_corr:
    corr_idx = torch.load(_corr_path, map_location=device, weights_only=False)
    print(f'  Loaded corr_idx from {_corr_path}')
else:
    if args.recompute_corr:
        print(f'  [INFO] Recomputing corr_idx (--recompute_corr)…')
    else:
        print(f'  [INFO] corr_idx.pt not found; computing on the fly…')
    from dataset.aircraft_damage import get_dataloader
    from codebase.models.mask_vae_aircraft import compute_skip_channel_correlations
    _DATA_ROOT  = './causal_data/aircraft damage'
    _tmp_loader = get_dataloader(_DATA_ROOT, 'valid', batch_size=64, num_workers=0)
    corr_idx = compute_skip_channel_correlations(lvae, _tmp_loader, device, top_k=32)
    torch.save(corr_idx, _corr_path)
    print(f'  Saved corr_idx → {_corr_path}')

clf = MultiLabelHead(z_dim=Z_DIM, n_classes=_N_OBS).to(device)
clf.load_state_dict(ckpt['clf'])
clf.eval()

sev_head = SeverityHead(z_dim=Z_DIM).to(device)
sev_head.load_state_dict(ckpt['sev_head'])
sev_head.eval()

dag_weights = lvae.dag.A.detach().cpu().numpy()

print('Checkpoint loaded:', args.checkpoint)
print(f'  Epoch: {ckpt.get("epoch", "?")}')
print(f'  Architecture: z1_dim={Z1_DIM}, z2_dim={Z2_DIM}, z_dim={Z_DIM}, n_observable={_N_OBS}')
print('\nLearned DAG edge weights (row=cause, col=effect):')
all_names = CLASS_NAMES[:Z1_DIM]
header = '             ' + '  '.join(f'{n[:9]:>9}' for n in all_names)
print(header)
for i, row in enumerate(dag_weights):
    vals = '  '.join(f'{v:9.3f}' for v in row)
    print(f'  {all_names[i][:12]:>12}  {vals}')

# ── Image preprocessing ───────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
_denorm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_denorm_std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_image(path: str) -> torch.Tensor:
    return _transform(Image.open(path).convert('RGB')).unsqueeze(0)


def to_display(tensor: torch.Tensor) -> np.ndarray:
    t = tensor.detach().cpu().float() * _denorm_std + _denorm_mean
    return (torch.clamp(t, 0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ── Forward pass helpers ──────────────────────────────────────────────────────
def _encode(img_tensor: torch.Tensor):
    """Encode image; return f_z1, z_dag, and skips for decoder."""
    with torch.no_grad():
        feat, skips = lvae.enc.encode(img_tensor)
        q_m_full, _ = ut.gaussian_parameters(feat, dim=1)   # (batch,64,1,1)
        q_m = lvae.enc_proj(q_m_full.view(img_tensor.size(0), -1))
        q_m = q_m.reshape([img_tensor.size(0), Z1_DIM, Z2_DIM])

        decode_m, _ = lvae.dag.calculate_dag(
            q_m,
            torch.ones(img_tensor.size(0), Z1_DIM, Z2_DIM).to(device),
        )
        decode_m = decode_m.reshape([img_tensor.size(0), Z1_DIM, Z2_DIM])
        m_zm  = lvae.dag.mask_z(decode_m).reshape([img_tensor.size(0), Z1_DIM, Z2_DIM])
        f_z   = lvae.mask_z.mix(m_zm).reshape([img_tensor.size(0), Z1_DIM, Z2_DIM])
        e_tilde = lvae.attn.attention(decode_m, q_m)[0]
        f_z1    = f_z + e_tilde
        z_dag   = f_z1 + torch.randn_like(f_z1) * 1e-4
    return f_z1, z_dag, skips


def _decode(z_dag: torch.Tensor, skips) -> torch.Tensor:
    """Decode z_dag using stored encoder skips."""
    with torch.no_grad():
        z_4d = z_dag.reshape([z_dag.size(0), Z_DIM, 1, 1])
        return lvae.dec.decode(z_4d, skips)


def _classify(z_dag: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = clf(z_dag.reshape(z_dag.size(0), -1))
        return torch.sigmoid(logits).cpu().numpy()[0]


def _predict_severity(z_dag: torch.Tensor) -> int:
    with torch.no_grad():
        pred = sev_head(z_dag.reshape(z_dag.size(0), -1)).item()
    return max(0, round(pred))


def _root_causes(z_dag: torch.Tensor) -> dict:
    """Extract latent root-cause probabilities from z_dag sub-vectors."""
    with torch.no_grad():
        rc = {}
        for i, name in enumerate(LATENT_NAMES):
            concept_idx = N_OBSERVABLE + i
            rc[name] = torch.sigmoid(z_dag[0, concept_idx, :].mean()).item()
    return rc


def grad_cam(img_tensor: torch.Tensor, concept_idx: int) -> np.ndarray:
    """
    Grad-CAM attribution heatmap for concept_idx (observable concepts only).
    """
    activations, gradients = {}, {}

    def fwd_hook(_module, _inp, out):
        activations['b'] = out

    def bwd_hook(_module, _grad_in, grad_out):
        gradients['b'] = grad_out[0]

    fh = lvae.enc.bottleneck.register_forward_hook(fwd_hook)
    bh = lvae.enc.bottleneck.register_full_backward_hook(bwd_hook)
    try:
        lvae.eval(); clf.eval()
        feat, _ = lvae.enc.encode(img_tensor)
        q_m_full, _ = ut.gaussian_parameters(feat, dim=1)
        q_m = lvae.enc_proj(q_m_full.view(1, -1)).reshape([1, Z1_DIM, Z2_DIM])
        decode_m, decode_v = lvae.dag.calculate_dag(
            q_m, torch.ones(1, Z1_DIM, Z2_DIM).to(device))
        decode_m = decode_m.reshape([1, Z1_DIM, Z2_DIM])
        m_zm    = lvae.dag.mask_z(decode_m).reshape([1, Z1_DIM, Z2_DIM])
        f_z     = lvae.mask_z.mix(m_zm).reshape([1, Z1_DIM, Z2_DIM])
        e_tilde = lvae.attn.attention(decode_m, q_m)[0]
        f_z1    = f_z + e_tilde
        logits  = clf(f_z1.reshape([1, Z_DIM]))
        logits[0, concept_idx].backward()

        A     = activations['b']                               # (1,256,4,4)
        G     = gradients['b']                                 # (1,256,4,4)
        alpha = G.mean(dim=(2, 3), keepdim=True)               # (1,256,1,1)
        cam   = F.relu((alpha * A).sum(dim=1, keepdim=True))   # (1,1,4,4)
        cam   = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE),
                              mode='bilinear', align_corners=False)
        cam   = cam[0, 0].detach().cpu().numpy()
        vmin, vmax = cam.min(), cam.max()
        if vmax > vmin:
            cam = (cam - vmin) / (vmax - vmin)
    except Exception as exc:
        print(f'  [WARN] grad_cam failed for concept {concept_idx}: {exc}')
        cam = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    finally:
        fh.remove()
        bh.remove()
    return cam


def apply_heatmap(orig_np: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Blend a Grad-CAM heatmap (jet colormap) onto an RGB uint8 image."""
    import matplotlib.cm as mcm
    heat = (mcm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    return (0.55 * orig_np + 0.45 * heat).astype(np.uint8)


def _intervention_probs(f_z1: torch.Tensor, concept_idx: int) -> np.ndarray:
    """
    Do-operator: set concept_idx sub-vector to -1 in f_z1, re-classify.
    Returns (_N_OBS,) probabilities — no image generation required.
    """
    with torch.no_grad():
        cf = f_z1.clone()
        cf[:, concept_idx, :] = -1.0
        return _classify(cf)


def suppress_skips(skips, concept_idx: int, strength: float = 1.0, n_suppress: int = 28):
    """
    Zero out the n_suppress most concept-correlated channels per skip level.
    """
    level_names = ['s1', 's2', 's3', 's4', 'b']
    result = []
    for name, s in zip(level_names, skips):
        if name in corr_idx:
            s_mod = s.clone()
            ch = corr_idx[name][concept_idx][:n_suppress]
            s_mod[:, ch] = s_mod[:, ch] * (1.0 - strength)
            result.append(s_mod)
        else:
            result.append(s)
    return tuple(result)


def concept_counterfactual(f_z1: torch.Tensor, concept_idx: int,
                           skips, strength: float = 1.0) -> torch.Tensor:
    """Skip-correlation counterfactual for observable concept concept_idx."""
    cf = f_z1.clone()
    cf[:, concept_idx, :] = -1.0
    skips_cf = suppress_skips(skips, concept_idx, strength)
    with torch.no_grad():
        z_4d = cf.reshape([1, Z_DIM, 1, 1])
        return lvae.dec.decode(z_4d, skips_cf)


# ── Helpers ───────────────────────────────────────────────────────────────────
def ascii_bar(probs: np.ndarray, width: int = 30) -> str:
    lines = []
    for i, p in enumerate(probs):
        bar = '#' * int(p * width)
        lines.append(f'  {OBS_NAMES[i]:>12s} [{bar:<{width}}] {p*100:5.1f}%')
    return '\n'.join(lines)


def root_cause_bar(name: str, val: float, width: int = 10) -> str:
    filled = int(val * width)
    empty  = width - filled
    blocks = '█' * filled + '░' * empty
    return f'  {name:<14s} {blocks} {val:.2f}'


def causal_notes(probs: np.ndarray, threshold: float = 0.5) -> list:
    detected = [OBS_NAMES[i] for i, p in enumerate(probs) if p >= threshold]
    notes = []
    for src in detected:
        downstream = [d for d in _CAUSAL_DOWNSTREAM.get(src, []) if d in detected]
        if downstream:
            notes.append(f'{src} → {", ".join(downstream)}  (causal chain confirmed by DAG)')
        src_idx = CLASS_NAMES.index(src)
        for j, tgt in enumerate(CLASS_NAMES[:Z1_DIM]):
            if j != src_idx and abs(float(dag_weights[src_idx, j])) > 0.3:
                notes.append(
                    f'  DAG edge {src} → {tgt}: weight={dag_weights[src_idx, j]:.3f}'
                )
    if not notes:
        notes.append('No strong causal chains detected.')
    return notes


def _prob_bars(ax, probs, ref_probs=None, show_delta=False):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(0, 1.4)
    ax.set_ylim(-0.5, _N_OBS - 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for i, p in enumerate(probs):
        y = _N_OBS - 1 - i
        ax.barh(y, float(p), height=0.55, color=OBS_COLORS[i])
        ax.text(-0.02, y, OBS_NAMES[i][:7], color='white',
                va='center', ha='right', fontsize=6)
        if show_delta and ref_probs is not None:
            delta = float(p) - float(ref_probs[i])
            sign  = '+' if delta >= 0 else ''
            dc    = '#2ecc71' if delta < -0.05 else ('#e74c3c' if delta > 0.05 else 'grey')
            ax.text(float(p) + 0.03, y, f'{sign}{delta:.2f}', color=dc,
                    va='center', fontsize=6)
        else:
            ax.text(float(p) + 0.03, y, f'{p*100:.0f}%', color='white',
                    va='center', fontsize=6)


def make_dag_figure(stem, dag_w, probs, sev_count, save_path):
    """7-node DAG diagram showing all concepts."""
    obs_nodes = [
        ('crack',     0.12, 0.55, 0),
        ('dent',      0.35, 0.55, 1),
        ('paint_off', 0.58, 0.55, 2),
        ('scratch',   0.82, 0.55, 3),
    ]
    latent_nodes = [
        ('impact_force',  0.20, 0.82, 4),
        ('metal_fatigue', 0.50, 0.82, 5),
        ('corrosion',     0.78, 0.82, 6),
    ]
    all_nodes = obs_nodes + latent_nodes
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(BG); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    for _, sx, sy, si in all_nodes:
        for _, dx, dy, di in all_nodes:
            if di == si:
                continue
            w = float(dag_w[si, di])
            if abs(w) < 0.05:
                continue
            lw    = max(0.5, min(5.0, abs(w) * 8))
            alpha = 0.85 if abs(w) > 0.3 else 0.45
            color = '#f1c40f' if abs(w) > 0.3 else '#555555'
            ax.annotate('', xy=(dx, dy), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=lw, alpha=alpha,
                                        connectionstyle='arc3,rad=0.08'))

    for name, x, y, idx in obs_nodes:
        detected = bool(probs[idx] >= args.threshold)
        color    = OBS_COLORS[idx] if detected else '#555555'
        ax.scatter([x], [y], s=1200, color=color, zorder=3,
                   edgecolors='white', linewidths=1.2)
        ax.text(x, y, name.replace('_', '\n'), ha='center', va='center',
                color='white', fontsize=6.5, fontweight='bold', zorder=4)

    for name, x, y, idx in latent_nodes:
        lc = LATENT_COLORS[idx - N_OBSERVABLE]
        ax.scatter([x], [y], s=1200, color=lc, zorder=3, marker='D',
                   edgecolors='white', linewidths=1.2)
        ax.text(x, y, name.replace('_', '\n'), ha='center', va='center',
                color='white', fontsize=5.5, fontweight='bold', zorder=4)

    ax.text(0.02, 0.04, '● observable', color='white', fontsize=7,
            transform=ax.transAxes)
    ax.text(0.18, 0.04, '◆ latent root cause', color='white', fontsize=7,
            transform=ax.transAxes)
    ax.set_title(f'Causal Structure — {stem}', color='white', fontsize=11, pad=10)
    ax.text(0.5, 0.04,
            f'Severity: {severity_label(sev_count)}',
            ha='center', color='#cccccc', fontsize=9, transform=ax.transAxes)

    plt.savefig(save_path, dpi=130, facecolor=BG, bbox_inches='tight')
    plt.close(fig)


def make_analysis_figure(stem, orig_img, recon_img, probs, root_causes,
                          dominant_cause, notes, sev_count, save_path):
    fig = plt.figure(figsize=(12, 7), facecolor=BG)
    fig.suptitle(f'Damage Analysis — {stem}', color='white', fontsize=10)
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 2],
                          hspace=0.15, wspace=0.25,
                          left=0.04, right=0.98, top=0.90, bottom=0.03)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(to_display(orig_img[0])); ax.set_title('Original', color='white', fontsize=9, pad=3)
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(to_display(recon_img[0])); ax.set_title('Reconstruction', color='white', fontsize=9, pad=3)
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 2])
    _prob_bars(ax, probs)
    ax.set_title('Damage Probabilities', color='white', fontsize=9, pad=3)

    # Root cause panel
    ax = fig.add_subplot(gs[0, 3])
    ax.set_facecolor(BG); ax.axis('off')
    ax.set_title('Root Causes', color='white', fontsize=9, pad=3)
    for i, (name, val) in enumerate(root_causes.items()):
        lc = LATENT_COLORS[i]
        bar_w = max(val, 0.01)
        ax.barh(2 - i, bar_w, height=0.5, color=lc, left=0)
        ax.text(bar_w + 0.02, 2 - i, f'{val:.2f}',
                color='white', va='center', fontsize=7)
        ax.text(-0.02, 2 - i, name.replace('_', '\n'),
                color='white', va='center', ha='right', fontsize=5.5)
    ax.set_xlim(0, 1.3); ax.set_ylim(-0.5, 2.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor(BG); ax.axis('off')
    for i, (name, p) in enumerate(zip(OBS_NAMES, probs)):
        marker = '●' if p >= args.threshold else '○'
        ax.text(0.05, 0.85 - i * 0.22, f'{marker} {name}: {p*100:.1f}%',
                color=OBS_COLORS[i], fontsize=9, transform=ax.transAxes)
    ax.set_title('Detections', color='white', fontsize=9)

    ax = fig.add_subplot(gs[1, 1:3])
    ax.set_facecolor(BG); ax.axis('off')
    sev_str  = severity_label(sev_count)
    note_str = '\n'.join(notes[:5])
    ax.text(0.02, 0.95,
            f'Severity: {sev_str}\n\nCausal Notes:\n{note_str}',
            color='#cccccc', fontsize=7, va='top',
            transform=ax.transAxes, wrap=True)

    ax = fig.add_subplot(gs[1, 3])
    ax.set_facecolor(BG); ax.axis('off')
    dc_color = LATENT_COLORS[list(root_causes.keys()).index(dominant_cause)]
    ax.text(0.05, 0.90, f'Dominant cause:', color='#aaaaaa', fontsize=7,
            transform=ax.transAxes)
    ax.text(0.05, 0.72, dominant_cause.replace('_', '\n'),
            color=dc_color, fontsize=9, fontweight='bold', transform=ax.transAxes)
    rec = _MAINTENANCE[dominant_cause]
    ax.text(0.05, 0.45, 'Recommendation:', color='#aaaaaa', fontsize=6,
            transform=ax.transAxes)
    ax.text(0.05, 0.05, rec, color='white', fontsize=5.5, va='bottom',
            transform=ax.transAxes, wrap=True)
    ax.set_title('Maintenance', color='white', fontsize=9)

    plt.savefig(save_path, dpi=130, facecolor=BG, bbox_inches='tight')
    plt.close(fig)


def make_causal_figure(stem, orig_img, cf_imgs, heatmaps, cf_probs,
                        ref_probs, save_path):
    """
    3-row × (1+N_OBSERVABLE)-col causal analysis figure.
      Row 1: Skip-suppressed counterfactual images
      Row 2: Grad-CAM attribution overlays
      Row 3: Causal intervention Δ probability bars
    """
    N = 1 + _N_OBS
    col_titles = ['original'] + [f'no_{n}' for n in OBS_NAMES]

    row_labels = [
        'Skip-suppressed counterfactual',
        'Grad-CAM attribution',
        'Causal intervention Δ',
    ]

    fig = plt.figure(figsize=(N * 2.4, 9.5), facecolor=BG)
    fig.suptitle(f'Causal Analysis — {stem}', color='white', fontsize=10)
    gs = fig.add_gridspec(3, N, height_ratios=[3, 3, 2],
                          hspace=0.18, wspace=0.12,
                          left=0.04, right=0.99, top=0.93, bottom=0.02)

    for c in range(N):
        ax = fig.add_subplot(gs[0, c])
        ax.set_facecolor(BG)
        if c == 0:
            ax.imshow(to_display(orig_img[0]))
            ax.set_title('original', color='white', fontsize=8, pad=3)
        else:
            ax.imshow(to_display(cf_imgs[c - 1][0]))
            ax.set_title(col_titles[c], color='white', fontsize=8, pad=3)
        ax.axis('off')
        if c == 0:
            ax.set_ylabel(row_labels[0], color='#aaaaaa', fontsize=7)

    for c in range(N):
        ax = fig.add_subplot(gs[1, c])
        ax.set_facecolor(BG)
        if c == 0:
            ax.imshow(to_display(orig_img[0]))
            ax.set_title('original', color='white', fontsize=8, pad=3)
        else:
            ax.imshow(heatmaps[c - 1])
            ax.set_title(f'{OBS_NAMES[c-1]} attr.', color='white', fontsize=8, pad=3)
        ax.axis('off')
        if c == 0:
            ax.set_ylabel(row_labels[1], color='#aaaaaa', fontsize=7)

    for c in range(N):
        ax = fig.add_subplot(gs[2, c])
        if c == 0:
            _prob_bars(ax, ref_probs)
            ax.set_title('baseline', color='white', fontsize=8, pad=3)
        else:
            _prob_bars(ax, cf_probs[c - 1], ref_probs=ref_probs, show_delta=True)
            ax.set_title(f'do({OBS_NAMES[c-1]}=0)', color='white', fontsize=8, pad=3)
        if c == 0:
            ax.set_ylabel(row_labels[2], color='#aaaaaa', fontsize=7)

    plt.savefig(save_path, dpi=130, facecolor=BG, bbox_inches='tight')
    plt.close(fig)


# ── Per-image analysis ────────────────────────────────────────────────────────
def analyse_image(img_path: str, out_dir: str) -> dict:
    stem       = Path(img_path).stem
    img_tensor = load_image(img_path).to(device)

    f_z1, z_dag, skips = _encode(img_tensor)
    recon_img           = _decode(z_dag, skips)
    ref_probs           = _classify(z_dag)
    sev_count           = _predict_severity(z_dag)
    root_causes         = _root_causes(z_dag)
    dominant_cause      = max(root_causes, key=root_causes.get)
    maintenance         = _MAINTENANCE[dominant_cause]

    # Skip-suppressed counterfactuals (observable concepts only)
    cf_imgs = [concept_counterfactual(f_z1, ci, skips) for ci in range(_N_OBS)]

    # Grad-CAM heatmaps (observable concepts only)
    orig_np  = to_display(img_tensor[0].cpu())
    cams     = [grad_cam(img_tensor, ci) for ci in range(_N_OBS)]
    heatmaps = [apply_heatmap(orig_np, cam) for cam in cams]

    # Causal intervention probability deltas
    cf_probs  = [_intervention_probs(f_z1, ci) for ci in range(_N_OBS)]
    notes = causal_notes(ref_probs, args.threshold)

    cf_causal_indicators = []
    for ci, name in enumerate(OBS_NAMES):
        indicators = {}
        for j, tgt in enumerate(OBS_NAMES):
            if j == ci:
                continue
            delta = float(cf_probs[ci][j] - ref_probs[j])
            if delta < -0.1:
                indicators[tgt] = 'causal_downstream_confirmed'
            elif delta > 0.1:
                indicators[tgt] = 'inverse_relationship'
            else:
                indicators[tgt] = 'no_change'
        cf_causal_indicators.append({name: indicators})

    # Console output
    print(f'\n{"="*60}\nImage: {img_path}')
    print('\nDamage probabilities:')
    print(ascii_bar(ref_probs))
    detected = [OBS_NAMES[i] for i, p in enumerate(ref_probs) if p >= args.threshold]
    print(f'\nDetected: {detected if detected else ["none"]}')
    print(f'Severity: {severity_label(sev_count)} (count={sev_count})')
    print('\nRoot causes:')
    for name, val in root_causes.items():
        print(root_cause_bar(name, val))
    print(f'  → Dominant: {dominant_cause}')
    print(f'  → {maintenance}')
    print('\nCausal notes:')
    for n in notes: print(f'  {n}')

    os.makedirs(out_dir, exist_ok=True)
    analysis_path = os.path.join(out_dir, f'{stem}_analysis.png')
    causal_path   = os.path.join(out_dir, f'{stem}_causal_analysis.png')
    dag_path      = os.path.join(out_dir, f'{stem}_causal_report.png')

    make_analysis_figure(stem, img_tensor.cpu(), recon_img.cpu(),
                         ref_probs, root_causes, dominant_cause,
                         notes, sev_count, analysis_path)
    make_causal_figure(stem, img_tensor.cpu(),
                       [cf.cpu() for cf in cf_imgs],
                       heatmaps, cf_probs, ref_probs, causal_path)
    make_dag_figure(stem, dag_weights, ref_probs, sev_count, dag_path)

    report = {
        'image':         img_path,
        'probabilities': {n: float(ref_probs[i]) for i, n in enumerate(OBS_NAMES)},
        'predictions':   {n: bool(ref_probs[i] >= args.threshold)
                          for i, n in enumerate(OBS_NAMES)},
        'damage_severity': {'count': sev_count, 'label': severity_label(sev_count)},
        'latent_root_causes': root_causes,
        'dominant_cause':     dominant_cause,
        'maintenance_recommendation': maintenance,
        'causal_notes':  notes,
        'counterfactual_deltas': {
            OBS_NAMES[ci]: {OBS_NAMES[j]: float(cf_probs[ci][j] - ref_probs[j])
                            for j in range(_N_OBS)}
            for ci in range(_N_OBS)
        },
        'causal_indicators': cf_causal_indicators,
        'dag_weights': {
            f'{CLASS_NAMES[i]}_to_{CLASS_NAMES[j]}': float(dag_weights[i, j])
            for i in range(Z1_DIM) for j in range(Z1_DIM)
            if abs(float(dag_weights[i, j])) > 0.05
        },
    }
    report_path = os.path.join(out_dir, f'{stem}_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f'  → {analysis_path}')
    print(f'  → {causal_path}')
    print(f'  → {dag_path}')
    print(f'  → {report_path}')
    return report


# ── Main ──────────────────────────────────────────────────────────────────────
if args.image is None and args.image_dir is None:
    parser.error('Provide --image or --image_dir')

image_paths = []
if args.image:
    image_paths.append(args.image)
if args.image_dir:
    exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_paths.extend(
        str(p) for p in sorted(Path(args.image_dir).iterdir())
        if p.suffix in exts
    )

if not image_paths:
    print('No images found.'); sys.exit(0)

all_reports = []
for img_path in image_paths:
    try:
        all_reports.append(analyse_image(img_path, args.out_dir))
    except Exception as e:
        print(f'ERROR processing {img_path}: {e}')
        import traceback; traceback.print_exc()

if len(all_reports) > 1:
    csv_path   = os.path.join(args.out_dir, 'summary.csv')
    fieldnames = (['image', 'severity_count', 'severity_label']
                  + [f'prob_{n}' for n in OBS_NAMES]
                  + [f'pred_{n}' for n in OBS_NAMES]
                  + [f'rc_{n}' for n in LATENT_NAMES]
                  + ['dominant_cause'])
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_reports:
            row = {
                'image':          r['image'],
                'severity_count': r['damage_severity']['count'],
                'severity_label': r['damage_severity']['label'],
                'dominant_cause': r['dominant_cause'],
            }
            for n in OBS_NAMES:
                row[f'prob_{n}'] = f"{r['probabilities'][n]:.4f}"
                row[f'pred_{n}'] = int(r['predictions'][n])
            for n in LATENT_NAMES:
                row[f'rc_{n}'] = f"{r['latent_root_causes'][n]:.4f}"
            writer.writerow(row)
    print(f'\nBatch summary → {csv_path}')

print(f'\nDone. {len(all_reports)} image(s) processed.')
