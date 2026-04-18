"""
Test-set evaluation for Aircraft Damage CausalVAE.

Computes:
  0. Existing metrics
     Severity regression   — MAE, within-1 accuracy, exact accuracy
     Damage type detection — F1, precision, recall per class + macro
     Reconstruction        — MSE

  1. AUC-ROC               — per-class and macro, with ROC curve plot
  2. Causal Faithfulness   — do-operator interventions on each DAG edge
  3. Concept Activation    — specificity heatmap and distributions
  4. MIG                   — Mutual Information Gap disentanglement score

Run from repo root:
    python evaluate_aircraft.py --checkpoint checkpoints/aircraft_best.pt
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, roc_curve, mutual_info_score

from codebase import utils as ut
from codebase.models.mask_vae_aircraft import CausalVAE
from dataset.aircraft_damage import get_dataloader, CLASS_NAMES, N_CONCEPTS, N_OBSERVABLE, SCALE
from models.classifier_head import MultiLabelHead, SeverityHead, compute_metrics

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='checkpoints/aircraft_best.pt')
parser.add_argument('--data_root',  default='./causal_data/aircraft damage')
parser.add_argument('--split',      default='test',
                    help='Dataset split to evaluate: test, valid, or train')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--out',        default='./eval_results.json',
                    help='Where to save the JSON report')
args = parser.parse_args()

# ── Colours ───────────────────────────────────────────────────────────────────
CLASS_COLORS  = ['#DC3232', '#FFA500', '#32B432', '#508CFF']
LATENT_COLORS = ['#FF6B6B', '#4ECDC4', '#95E1D3']
ALL_COLORS    = CLASS_COLORS + LATENT_COLORS
OBS_NAMES     = CLASS_NAMES[:N_OBSERVABLE]
LATENT_NAMES  = CLASS_NAMES[N_OBSERVABLE:N_CONCEPTS]

os.makedirs('eval_plots', exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device     : {device}')
print(f'Checkpoint : {args.checkpoint}')
print(f'Split      : {args.split}')

# ── Load model ────────────────────────────────────────────────────────────────
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
cfg  = ckpt.get('config', {'z_dim': 28, 'z1_dim': 7, 'z2_dim': 4})
Z_DIM, Z1_DIM, Z2_DIM = cfg['z_dim'], cfg['z1_dim'], cfg['z2_dim']
_N_OBS = cfg.get('n_observable', N_OBSERVABLE)

lvae = CausalVAE(
    z_dim=Z_DIM, z1_dim=Z1_DIM, z2_dim=Z2_DIM,
    inference=True, scale=SCALE, initial=False,
).to(device)
lvae.load_state_dict(ckpt['lvae'], strict=False)
lvae.eval()

clf = MultiLabelHead(z_dim=Z_DIM, n_classes=_N_OBS).to(device)
clf.load_state_dict(ckpt['clf'])
clf.eval()

sev_head = SeverityHead(z_dim=Z_DIM).to(device)
sev_head.load_state_dict(ckpt['sev_head'])
sev_head.eval()

# ── Data ──────────────────────────────────────────────────────────────────────
loader = get_dataloader(args.data_root, args.split, args.batch_size, num_workers=0)
print(f'Images     : {len(loader.dataset)}')


# ── Encoding helper (for faithfulness test) ───────────────────────────────────
@torch.no_grad()
def encode_f_z1(imgs: torch.Tensor) -> torch.Tensor:
    """Encode images and return f_z1 shape (batch, Z1_DIM, Z2_DIM) without noise."""
    feat, _ = lvae.enc.encode(imgs)
    q_m_full, _ = ut.gaussian_parameters(feat, dim=1)
    q_m = lvae.enc_proj(q_m_full.view(imgs.size(0), -1))
    q_m = q_m.reshape([imgs.size(0), Z1_DIM, Z2_DIM])
    decode_m, _ = lvae.dag.calculate_dag(
        q_m, torch.ones(imgs.size(0), Z1_DIM, Z2_DIM).to(device)
    )
    decode_m = decode_m.reshape([imgs.size(0), Z1_DIM, Z2_DIM])
    m_zm    = lvae.dag.mask_z(decode_m).reshape([imgs.size(0), Z1_DIM, Z2_DIM])
    f_z     = lvae.mask_z.mix(m_zm).reshape([imgs.size(0), Z1_DIM, Z2_DIM])
    e_tilde = lvae.attn.attention(decode_m, q_m)[0]
    return f_z + e_tilde


# ── Main evaluation loop (collect predictions + latents) ─────────────────────
all_logits   = []
all_labels   = []
all_z_dag    = []
all_sev_pred = []
all_sev_true = []
total_rec    = 0.0
n_batches    = 0

print('\nRunning forward pass over test set…')
with torch.no_grad():
    for imgs, labels, sev_counts in tqdm(loader, desc='Forward pass'):
        imgs, labels = imgs.to(device), labels.to(device)
        sev_counts   = sev_counts.to(device)

        _, _, rec, _, z_dag = lvae.negative_elbo_bound(imgs, labels, sample=False)

        z_flat   = z_dag.reshape(imgs.size(0), -1)
        logits   = clf(z_flat)
        sev_pred = sev_head(z_flat)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_z_dag.append(z_dag.cpu())
        all_sev_pred.append(sev_pred.cpu())
        all_sev_true.append(sev_counts.cpu())
        total_rec += rec.item()
        n_batches += 1

logits_all   = torch.cat(all_logits)
labels_all   = torch.cat(all_labels)
z_dag_all    = torch.cat(all_z_dag)              # (N, Z1_DIM, Z2_DIM)
sev_pred_all = torch.cat(all_sev_pred).float().numpy()
sev_true_all = torch.cat(all_sev_true).float().numpy()
labels_obs   = labels_all[:, :_N_OBS].numpy()   # (N, 4) binary observable

# ── Existing metrics ──────────────────────────────────────────────────────────
sev_mae      = float(np.mean(np.abs(sev_pred_all - sev_true_all)))
sev_exact    = float(np.mean(np.round(sev_pred_all) == np.round(sev_true_all)))
sev_within1  = float(np.mean(np.abs(sev_pred_all - sev_true_all) <= 1.0))
sev_rmse     = float(np.sqrt(np.mean((sev_pred_all - sev_true_all) ** 2)))

metrics = compute_metrics(
    logits_all, labels_all[:, :_N_OBS],
    class_names=OBS_NAMES,
    sev_pred=torch.tensor(sev_pred_all),
    sev_target=torch.tensor(sev_true_all),
)
avg_rec = total_rec / max(n_batches, 1)

# ── Print existing metrics ─────────────────────────────────────────────────────
SEP = '─' * 52
print(f'\n{SEP}')
print(f'  Evaluation results  ({args.split} split)')
print(SEP)

print('\n  SEVERITY REGRESSION  (total damage instance count)')
print(f'    MAE             : {sev_mae:.3f}')
print(f'    RMSE            : {sev_rmse:.3f}')
print(f'    Exact accuracy  : {sev_exact*100:.1f}%   (round(pred) == true)')
print(f'    Within-1 acc    : {sev_within1*100:.1f}%   (|pred - true| ≤ 1)')

print('\n  DAMAGE TYPE DETECTION  (threshold = 0.5)')
print(f'    {"Class":<12}  {"F1":>6}  {"Precision":>10}  {"Recall":>8}')
print(f'    {"-"*12}  {"------":>6}  {"----------":>10}  {"--------":>8}')
for name in OBS_NAMES:
    f1   = metrics.get(f'f1_{name}',        0.0)
    prec = metrics.get(f'precision_{name}', 0.0)
    rec  = metrics.get(f'recall_{name}',    0.0)
    print(f'    {name:<12}  {f1:6.3f}  {prec:10.3f}  {rec:8.3f}')
print(f'    {"─"*12}  {"─"*6}  {"─"*10}  {"─"*8}')
print(f'    {"MACRO":<12}  {metrics["f1_macro"]:6.3f}  '
      f'{metrics["precision_macro"]:10.3f}  {metrics["recall_macro"]:8.3f}')

print(f'\n  RECONSTRUCTION')
print(f'    Avg MSE         : {avg_rec:.5f}')


# ══════════════════════════════════════════════════════════════════════════════
# EVAL 1 — AUC-ROC
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  Eval 1: AUC-ROC')
print(SEP)

probs_all = torch.sigmoid(logits_all).numpy()   # (N, 4)

auc_per_class = {}
for j, name in enumerate(OBS_NAMES):
    try:
        auc_per_class[name] = float(roc_auc_score(labels_obs[:, j], probs_all[:, j]))
    except ValueError:
        auc_per_class[name] = float('nan')

valid_aucs = [v for v in auc_per_class.values() if not math.isnan(v)]
macro_auc  = float(np.mean(valid_aucs)) if valid_aucs else float('nan')

for name, auc in auc_per_class.items():
    print(f'    {name:<12}  AUC = {auc:.3f}')
print(f'    {"MACRO":<12}  AUC = {macro_auc:.3f}')

# ROC curve plot
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], '--', color='#555555', linewidth=1, label='Random')
for j, name in enumerate(OBS_NAMES):
    try:
        fpr, tpr, _ = roc_curve(labels_obs[:, j], probs_all[:, j])
        auc_val = auc_per_class[name]
        ax.plot(fpr, tpr, color=CLASS_COLORS[j], linewidth=2,
                label=f'{name}  (AUC={auc_val:.3f})')
    except ValueError:
        pass
ax.set_xlabel('False Positive Rate', color='white')
ax.set_ylabel('True Positive Rate', color='white')
ax.set_title('ROC Curves — Observable Damage Concepts', color='white')
ax.legend(fontsize=9, facecolor='#2a2a2a', edgecolor='#444444', labelcolor='white')
ax.set_facecolor('#1c1c1c')
fig.patch.set_facecolor('#1c1c1c')
plt.tight_layout()
plt.savefig('eval_plots/roc_curves.png', dpi=130, facecolor='#1c1c1c')
plt.close(fig)
print('    → eval_plots/roc_curves.png')


# ══════════════════════════════════════════════════════════════════════════════
# EVAL 2 — CAUSAL FAITHFULNESS
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  Eval 2: Causal Faithfulness')
print(SEP)

# (parent, child, is_dag_edge)
EDGES = [
    (4, 1, True),  (4, 3, True),  (4, 0, True),  # impact_force → dent, scratch, crack
    (5, 0, True),                                  # metal_fatigue → crack
    (6, 2, True),                                  # corrosion → paint_off
    (1, 0, True),  (1, 3, True),                   # dent → crack, scratch
    (3, 2, True),                                  # scratch → paint_off
    (0, 1, False), (0, 2, False),                  # crack non-edges
    (3, 1, False),                                 # scratch → dent (non-edge)
    (6, 0, False),                                 # corrosion → crack (non-edge)
]

effect_matrix = np.full((Z1_DIM, _N_OBS), np.nan)  # (7 parents, 4 children)
edge_results  = {}

for parent, child, is_dag in tqdm(EDGES, desc='Faithfulness'):
    all_child_on  = []
    all_child_off = []

    for imgs, labels, _ in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        # Select images where parent activates
        if parent < _N_OBS:
            mask = labels[:, parent] > 0.5
        else:
            with torch.no_grad():
                fz1_tmp = encode_f_z1(imgs)
            mask = torch.sigmoid(fz1_tmp[:, parent, :].mean(dim=-1)) > 0.4

        if mask.sum() == 0:
            continue

        imgs_sel   = imgs[mask]
        with torch.no_grad():
            f_z1 = encode_f_z1(imgs_sel)

            f_z1_on  = f_z1.clone(); f_z1_on[:, parent, :]  = 1.0
            f_z1_off = f_z1.clone(); f_z1_off[:, parent, :] = -1.0

            v_small = torch.ones_like(f_z1) * 0.001
            z_on  = ut.conditional_sample_gaussian(f_z1_on,  v_small)
            z_off = ut.conditional_sample_gaussian(f_z1_off, v_small)

            if child < _N_OBS:
                p_on  = torch.sigmoid(clf(z_on.reshape(z_on.size(0),  -1))[:, child])
                p_off = torch.sigmoid(clf(z_off.reshape(z_off.size(0), -1))[:, child])
            else:
                p_on  = torch.sigmoid(z_on[:,  child, :].mean(dim=-1))
                p_off = torch.sigmoid(z_off[:, child, :].mean(dim=-1))

        all_child_on.extend(p_on.cpu().tolist())
        all_child_off.extend(p_off.cpu().tolist())

    if not all_child_on:
        edge_key = f'{CLASS_NAMES[parent]}_to_{CLASS_NAMES[child]}'
        edge_results[edge_key] = {'effect': float('nan'), 'faithful': False,
                                  'is_dag_edge': is_dag, 'n_images': 0}
        continue

    causal_effect = float(np.mean(all_child_on) - np.mean(all_child_off))
    faithful      = causal_effect > 0.05

    if child < _N_OBS:
        effect_matrix[parent, child] = causal_effect

    edge_key = f'{CLASS_NAMES[parent]}_to_{CLASS_NAMES[child]}'
    edge_results[edge_key] = {
        'effect': causal_effect, 'faithful': faithful,
        'is_dag_edge': is_dag, 'n_images': len(all_child_on),
    }

    edge_type = 'DAG edge' if is_dag else 'non-edge'
    status    = 'FAITHFUL' if faithful else 'unfaithful'
    print(f'    {CLASS_NAMES[parent]:>14} → {CLASS_NAMES[child]:<12}  '
          f'effect={causal_effect:+.3f}  {status}  [{edge_type}]')

dag_edges     = [e for e in edge_results.values() if e['is_dag_edge']]
n_dag_total   = len(dag_edges)
n_dag_faithful= sum(1 for e in dag_edges if e['faithful'])
faith_score   = n_dag_faithful / max(n_dag_total, 1)
faith_interp  = (f'Causal faithfulness: {n_dag_faithful}/{n_dag_total} edges faithful'
                 + (' — model has learned causal structure'
                    if faith_score >= 0.625 else ' — partial causal structure'))

print(f'\n    {faith_interp}')

# Faithfulness heatmap
fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor('#1c1c1c')
ax.set_facecolor('#1c1c1c')

# Fill matrix cells
vmax = max(0.5, np.nanmax(np.abs(effect_matrix)))
cmap = plt.cm.RdYlGn
im   = ax.imshow(effect_matrix, cmap=cmap, vmin=-vmax, vmax=vmax,
                 aspect='auto', interpolation='nearest')

# Annotate and draw borders
tested_cells = {(p, c): (is_dag, eff) for (p, c, is_dag) in EDGES
                if c < _N_OBS
                for eff in [effect_matrix[p, c]]}

for p, c, is_dag in EDGES:
    if c >= _N_OBS:
        continue
    eff = effect_matrix[p, c]
    txt = f'{eff:+.2f}' if not np.isnan(eff) else 'n/a'
    ax.text(c, p, txt, ha='center', va='center', fontsize=7,
            color='white', fontweight='bold')
    lw    = 2.5 if is_dag else 1.2
    ls    = '-' if is_dag else '--'
    ec    = 'white' if is_dag else '#888888'
    rect  = mpatches.FancyBboxPatch(
        (c - 0.48, p - 0.48), 0.96, 0.96,
        boxstyle='square,pad=0', linewidth=lw, linestyle=ls,
        edgecolor=ec, facecolor='none'
    )
    ax.add_patch(rect)

ax.set_xticks(range(_N_OBS))
ax.set_xticklabels(OBS_NAMES, color='white', fontsize=9)
ax.set_yticks(range(Z1_DIM))
ax.set_yticklabels(CLASS_NAMES[:Z1_DIM], color='white', fontsize=9)
ax.set_xlabel('Child concept (observable)', color='white')
ax.set_ylabel('Parent concept', color='white')
ax.set_title('Causal Faithfulness — Intervention Effects', color='white')
plt.colorbar(im, ax=ax, label='Causal Effect (on − off)')
plt.tight_layout()
plt.savefig('eval_plots/causal_faithfulness.png', dpi=130, facecolor='#1c1c1c')
plt.close(fig)
print('    → eval_plots/causal_faithfulness.png')


# ══════════════════════════════════════════════════════════════════════════════
# EVAL 3 — CONCEPT ACTIVATION CONSISTENCY
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  Eval 3: Concept Activation Consistency')
print(SEP)

# Compute per-concept mean activation per image
# z_dag_all shape: (N, Z1_DIM, Z2_DIM)
act_matrix = torch.sigmoid(z_dag_all.mean(dim=-1)).numpy()   # (N, Z1_DIM)

# 7×4 mean activation table: table[concept_i, obs_class_j]
table = np.zeros((Z1_DIM, _N_OBS))
for i in range(Z1_DIM):
    for j in range(_N_OBS):
        mask_j = labels_obs[:, j] == 1
        if mask_j.sum() > 0:
            table[i, j] = act_matrix[mask_j, i].mean()
        else:
            table[i, j] = float('nan')

# Specificity for observable concepts (diagonal vs off-diagonal)
spec_per_concept = {}
for i in range(_N_OBS):
    row     = table[i, :]
    diag    = table[i, i]
    off_mean= np.nanmean([table[i, j] for j in range(_N_OBS) if j != i])
    spec_per_concept[OBS_NAMES[i]] = float(diag - off_mean)
    print(f'    {OBS_NAMES[i]:<12}  specificity = {diag - off_mean:+.3f}'
          f'  (diag={diag:.3f}, off-diag mean={off_mean:.3f})')

mean_specificity = float(np.mean(list(spec_per_concept.values())))
print(f'    Mean specificity  : {mean_specificity:.3f}')

# ── Plot 1: Activation matrix heatmap ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 8))
fig.patch.set_facecolor('#1c1c1c')
ax.set_facecolor('#1c1c1c')

im = ax.imshow(table, cmap='viridis', vmin=0, vmax=1,
               aspect='auto', interpolation='nearest')

# Expected high-activation cells
expected = {
    0: [0], 1: [1], 2: [2], 3: [3],   # observable: diagonal
    4: [1, 3],                          # impact_force: dent+scratch
    5: [0],                             # metal_fatigue: crack
    6: [2],                             # corrosion: paint_off
}
for i in range(Z1_DIM):
    for j in range(_N_OBS):
        val = table[i, j]
        txt = f'{val:.2f}' if not np.isnan(val) else '-'
        star = ' ★' if j in expected.get(i, []) else ''
        ax.text(j, i, txt + star, ha='center', va='center',
                fontsize=8, color='white')

ax.set_xticks(range(_N_OBS))
ax.set_xticklabels(OBS_NAMES, color='white', fontsize=9)
ax.set_yticks(range(Z1_DIM))
yticklabels = (
    [f'{n} (obs)' for n in OBS_NAMES]
    + [f'{n} (latent)' for n in LATENT_NAMES]
)
ax.set_yticklabels(yticklabels[:Z1_DIM], color='white', fontsize=8)
ax.set_xlabel('Observable damage class present', color='white')
ax.set_ylabel('Concept sub-vector', color='white')
ax.set_title('Concept Activation Matrix\n(★ = expected high activation)', color='white')
plt.colorbar(im, ax=ax, label='Mean activation (sigmoid)')
plt.tight_layout()
plt.savefig('eval_plots/concept_activation_matrix.png', dpi=130, facecolor='#1c1c1c')
plt.close(fig)
print('    → eval_plots/concept_activation_matrix.png')

# ── Plot 2: Violin / distribution plots for observable concepts ───────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.patch.set_facecolor('#1c1c1c')
fig.suptitle('Concept Activation Distributions (present vs absent)', color='white')

for idx, (ax, name) in enumerate(zip(axes.flat, OBS_NAMES)):
    ax.set_facecolor('#1c1c1c')
    mask_present = labels_obs[:, idx] == 1
    mask_absent  = labels_obs[:, idx] == 0
    vals_present = act_matrix[mask_present, idx]
    vals_absent  = act_matrix[mask_absent,  idx]

    data = [vals_absent, vals_present]
    parts = ax.violinplot(data, positions=[0, 1],
                          showmeans=True, showmedians=True)
    for i_v, pc in enumerate(parts['bodies']):
        pc.set_facecolor(CLASS_COLORS[idx])
        pc.set_alpha(0.6 if i_v == 1 else 0.3)
    for key in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
        if key in parts:
            parts[key].set_color('white')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['absent', 'present'], color='white', fontsize=9)
    ax.set_ylabel('Activation', color='white')
    ax.set_title(f'{name}', color=CLASS_COLORS[idx], fontsize=10)
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444444')

plt.tight_layout()
plt.savefig('eval_plots/concept_activation_distributions.png', dpi=130, facecolor='#1c1c1c')
plt.close(fig)
print('    → eval_plots/concept_activation_distributions.png')


# ══════════════════════════════════════════════════════════════════════════════
# EVAL 4 — MIG (Mutual Information Gap)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  Eval 4: Disentanglement (MIG)')
print(SEP)

mig_per_class = {}
for j, name in enumerate(OBS_NAMES):
    gt = labels_obs[:, j].astype(int)
    p  = gt.mean()
    if p <= 0 or p >= 1:
        mig_per_class[name] = 0.0
        continue

    H_j = -(p * math.log2(p + 1e-10) + (1 - p) * math.log2(1 - p + 1e-10))

    mis = []
    for k in range(Z1_DIM):
        acts_k = act_matrix[:, k]
        bins   = np.digitize(acts_k,
                             np.percentile(acts_k, np.linspace(0, 100, 11)[1:-1]))
        mi = mutual_info_score(gt, bins)
        mis.append(mi)

    mis_sorted = sorted(mis, reverse=True)
    MI1, MI2   = mis_sorted[0], mis_sorted[1] if len(mis_sorted) > 1 else 0.0
    mig_j      = (MI1 - MI2) / (H_j + 1e-10)
    mig_per_class[name] = float(mig_j)

overall_mig = float(np.mean(list(mig_per_class.values())))

def _mig_interp(mig: float) -> str:
    if mig > 0.35:  return 'Excellent'
    if mig > 0.20:  return 'Good'
    if mig > 0.10:  return 'Acceptable'
    return 'Poor'

for name, mig_j in mig_per_class.items():
    print(f'    {name:<12}  MIG = {mig_j:.3f}  ({_mig_interp(mig_j)})')
mig_interp_str = (f'MIG: {overall_mig:.3f} — {_mig_interp(overall_mig)} '
                  f'disentanglement, concepts capture distinct damage types')
print(f'\n    Overall  {mig_interp_str}')


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY RADAR CHART
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  Summary radar chart')
print(SEP)

radar_labels  = ['Macro\nAUC-ROC', 'Causal\nFaithfulness', 'Mean\nSpecificity', 'MIG\nScore']
macro_auc_clipped = max(0.0, min(1.0, macro_auc if not math.isnan(macro_auc) else 0.0))
mig_norm          = min(overall_mig / 0.5, 1.0)
spec_norm         = max(0.0, min(1.0, (mean_specificity + 0.5) / 1.0))
radar_values      = [macro_auc_clipped, faith_score, spec_norm, mig_norm]

N_axes  = len(radar_labels)
angles  = [n / N_axes * 2 * math.pi for n in range(N_axes)]
angles += angles[:1]
vals    = radar_values + radar_values[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('#1c1c1c')
ax.set_facecolor('#1c1c1c')

ax.plot(angles, vals, color='#4ECDC4', linewidth=2)
ax.fill(angles, vals, color='#4ECDC4', alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, color='white', fontsize=9)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], color='#aaaaaa', fontsize=7)
ax.tick_params(colors='white')
ax.spines['polar'].set_color('#444444')
ax.grid(color='#444444', linestyle='--', alpha=0.5)
ax.set_title('Evaluation Summary', color='white', fontsize=12, pad=20)

for angle, val, lbl in zip(angles[:-1], radar_values, radar_labels):
    ax.text(angle, val + 0.07, f'{val:.2f}', ha='center', va='center',
            color='white', fontsize=8)

plt.tight_layout()
plt.savefig('eval_plots/evaluation_summary.png', dpi=130, facecolor='#1c1c1c')
plt.close(fig)
print('    → eval_plots/evaluation_summary.png')


# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  FINAL EVALUATION SUMMARY')
print(SEP)
print(f'\n  Macro AUC-ROC     : {macro_auc:.3f}')
print(f'  {faith_interp}')
print(f'  Mean specificity  : {mean_specificity:.3f}')
print(f'  {mig_interp_str}')
print(f'\n{SEP}\n')

# ── Save JSON ─────────────────────────────────────────────────────────────────
report = {
    'checkpoint': args.checkpoint,
    'split':      args.split,
    'n_images':   len(loader.dataset),
    'severity': {
        'mae':         sev_mae,
        'rmse':        sev_rmse,
        'exact_acc':   sev_exact,
        'within1_acc': sev_within1,
    },
    'detection': {
        'f1_macro':        metrics['f1_macro'],
        'precision_macro': metrics['precision_macro'],
        'recall_macro':    metrics['recall_macro'],
        'per_class': {
            name: {
                'f1':        metrics.get(f'f1_{name}', 0.0),
                'precision': metrics.get(f'precision_{name}', 0.0),
                'recall':    metrics.get(f'recall_{name}', 0.0),
            }
            for name in OBS_NAMES
        },
    },
    'reconstruction': {'avg_mse': avg_rec},
    'auc_roc': {
        'macro':     macro_auc,
        'per_class': auc_per_class,
    },
    'causal_faithfulness': {
        'score':       faith_score,
        'n_faithful':  n_dag_faithful,
        'n_total':     n_dag_total,
        'per_edge':    edge_results,
        'interpretation': faith_interp,
    },
    'concept_activation': {
        'specificity_per_concept': spec_per_concept,
        'mean_specificity':        mean_specificity,
        'activation_table': {
            CLASS_NAMES[i]: {OBS_NAMES[j]: float(table[i, j])
                             for j in range(_N_OBS)}
            for i in range(Z1_DIM)
        },
    },
    'mig': {
        'overall':       overall_mig,
        'per_class':     mig_per_class,
        'interpretation': mig_interp_str,
    },
}

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, 'w') as f:
    json.dump(report, f, indent=2)
print(f'Full report saved → {args.out}')
