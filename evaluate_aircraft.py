"""
Test-set evaluation for Aircraft Damage CausalVAE.

Computes:
  0. Existing metrics
     Severity regression   — MAE, within-1 accuracy, exact accuracy
     Damage type detection — F1, precision, recall per class + macro
     Reconstruction        — MSE

  1. AUC-ROC               — per-class and macro, with ROC curve plot
  2. Concept Activation    — specificity heatmap and distributions
  3. MIC / TIC             — Mutual Information Completeness / Total Information Content

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
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, roc_curve, mutual_info_score

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
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], '--', color='#aaaaaa', linewidth=1, label='Random')
for j, name in enumerate(OBS_NAMES):
    try:
        fpr, tpr, _ = roc_curve(labels_obs[:, j], probs_all[:, j])
        auc_val = auc_per_class[name]
        ax.plot(fpr, tpr, color=CLASS_COLORS[j], linewidth=2,
                label=f'{name}  (AUC={auc_val:.3f})')
    except ValueError:
        pass
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Observable Damage Concepts')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('eval_plots/roc_curves.png', dpi=130)
plt.close(fig)
print('    → eval_plots/roc_curves.png')




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
                fontsize=8, color='black')

ax.set_xticks(range(_N_OBS))
ax.set_xticklabels(OBS_NAMES, fontsize=9)
ax.set_yticks(range(Z1_DIM))
yticklabels = (
    [f'{n} (obs)' for n in OBS_NAMES]
    + [f'{n} (latent)' for n in LATENT_NAMES]
)
ax.set_yticklabels(yticklabels[:Z1_DIM], fontsize=8)
ax.set_xlabel('Observable damage class present')
ax.set_ylabel('Concept sub-vector')
ax.set_title('Concept Activation Matrix\n(★ = expected high activation)')
plt.colorbar(im, ax=ax, label='Mean activation (sigmoid)')
plt.tight_layout()
plt.savefig('eval_plots/concept_activation_matrix.png', dpi=130)
plt.close(fig)
print('    → eval_plots/concept_activation_matrix.png')

# ── Plot 2: Violin / distribution plots for observable concepts ───────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.suptitle('Concept Activation Distributions (present vs absent)')

for idx, (ax, name) in enumerate(zip(axes.flat, OBS_NAMES)):
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
            parts[key].set_color('black')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['absent', 'present'], fontsize=9)
    ax.set_ylabel('Activation')
    ax.set_title(f'{name}', color=CLASS_COLORS[idx], fontsize=10)
    for sp in ax.spines.values():
        sp.set_edgecolor('#cccccc')

plt.tight_layout()
plt.savefig('eval_plots/concept_activation_distributions.png', dpi=130)
plt.close(fig)
print('    → eval_plots/concept_activation_distributions.png')


# ══════════════════════════════════════════════════════════════════════════════
# EVAL 4 — MIC / TIC (Mutual Information Completeness / Total Information Content)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  Eval 4: Disentanglement (MIC / TIC)')
print(SEP)

# MI matrix: mi_matrix[i, j] = MI(activation of z_i, label g_j)  shape (Z1_DIM, _N_OBS)
mi_matrix = np.zeros((Z1_DIM, _N_OBS))
for i in range(Z1_DIM):
    acts_i = act_matrix[:, i]
    bins_i = np.digitize(acts_i, np.percentile(acts_i, np.linspace(0, 100, 11)[1:-1]))
    for j in range(_N_OBS):
        mi_matrix[i, j] = mutual_info_score(labels_obs[:, j].astype(int), bins_i)

# MIC — completeness: for each ground-truth factor, best-matching latent
mic_per_class = {OBS_NAMES[j]: float(np.max(mi_matrix[:, j])) for j in range(_N_OBS)}
overall_mic   = float(np.mean(list(mic_per_class.values())))

# TIC — utility: for each latent, best-matching ground-truth factor
tic_per_concept = {CLASS_NAMES[i]: float(np.max(mi_matrix[i, :])) for i in range(Z1_DIM)}
overall_tic     = float(np.mean(list(tic_per_concept.values())))

# Normalise against mean binary entropy of labels (nats) for radar chart
h_vals = []
for j in range(_N_OBS):
    p = labels_obs[:, j].mean()
    if 0 < p < 1:
        h_vals.append(-(p * math.log(p) + (1 - p) * math.log(1 - p)))
h_mean  = float(np.mean(h_vals)) if h_vals else 0.693
mic_norm = min(overall_mic / h_mean, 1.0)
tic_norm = min(overall_tic / h_mean, 1.0)

def _mi_interp(mi: float) -> str:
    if mi > 0.30:  return 'Excellent'
    if mi > 0.15:  return 'Good'
    if mi > 0.05:  return 'Acceptable'
    return 'Poor'

print('  MIC (coverage — each ground-truth factor captured by best latent):')
for name, mic_j in mic_per_class.items():
    print(f'    {name:<12}  MIC = {mic_j:.3f}  ({_mi_interp(mic_j)})')
print(f'    Overall MIC = {overall_mic:.3f}  ({_mi_interp(overall_mic)})')

print('\n  TIC (utility — each latent captures something meaningful):')
for name, tic_i in tic_per_concept.items():
    print(f'    {name:<16}  TIC = {tic_i:.3f}  ({_mi_interp(tic_i)})')
print(f'    Overall TIC = {overall_tic:.3f}  ({_mi_interp(overall_tic)})')

mic_interp_str = f'MIC: {overall_mic:.3f} — {_mi_interp(overall_mic)} coverage of ground-truth factors'
tic_interp_str = f'TIC: {overall_tic:.3f} — {_mi_interp(overall_tic)} latent concept utility'


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY RADAR CHART
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  Summary radar chart')
print(SEP)

radar_labels  = ['Macro\nAUC-ROC', 'Mean\nSpecificity', 'MIC', 'TIC']
macro_auc_clipped = max(0.0, min(1.0, macro_auc if not math.isnan(macro_auc) else 0.0))
spec_norm         = max(0.0, min(1.0, (mean_specificity + 0.5) / 1.0))
radar_values      = [macro_auc_clipped, spec_norm, mic_norm, tic_norm]

N_axes  = len(radar_labels)
angles  = [n / N_axes * 2 * math.pi for n in range(N_axes)]
angles += angles[:1]
vals    = radar_values + radar_values[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

ax.plot(angles, vals, color='#4ECDC4', linewidth=2)
ax.fill(angles, vals, color='#4ECDC4', alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=9)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], color='#666666', fontsize=7)
ax.spines['polar'].set_color('#cccccc')
ax.grid(color='#dddddd', linestyle='--', alpha=0.5)
ax.set_title('Evaluation Summary', fontsize=12, pad=20)

for angle, val, lbl in zip(angles[:-1], radar_values, radar_labels):
    ax.text(angle, val + 0.07, f'{val:.2f}', ha='center', va='center',
            fontsize=8)

plt.tight_layout()
plt.savefig('eval_plots/evaluation_summary.png', dpi=130)
plt.close(fig)
print('    → eval_plots/evaluation_summary.png')


# ══════════════════════════════════════════════════════════════════════════════
# DAG STRUCTURE DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
# Topological layers: latent roots → observable intermediates → observable sinks
# dent and scratch cause other observable nodes, so they sit in the middle row
_dag_nodes = {
    # idx: (label, x, y)
    4: ('impact_force',  0.20, 0.82),
    5: ('metal_fatigue', 0.50, 0.82),
    6: ('corrosion',     0.80, 0.82),
    1: ('dent',          0.35, 0.50),
    3: ('scratch',       0.65, 0.50),
    0: ('crack',         0.35, 0.18),
    2: ('paint_off',     0.65, 0.18),
}
_dag_edges = [
    (4, 1), (4, 3), (4, 0),
    (5, 0),
    (6, 2),
    (1, 0), (1, 3),
    (3, 2),
]

fig, ax = plt.subplots(figsize=(7, 8))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# Row labels on the right
ax.text(0.97, 0.82, 'latent\nroot causes',     color='#888888', fontsize=7,
        va='center', ha='right', transform=ax.transAxes)
ax.text(0.97, 0.50, 'observable\n(intermediate)', color='#888888', fontsize=7,
        va='center', ha='right', transform=ax.transAxes)
ax.text(0.97, 0.18, 'observable\n(sink)',         color='#888888', fontsize=7,
        va='center', ha='right', transform=ax.transAxes)

for src_i, dst_i in _dag_edges:
    sx, sy = _dag_nodes[src_i][1], _dag_nodes[src_i][2]
    dx, dy = _dag_nodes[dst_i][1], _dag_nodes[dst_i][2]
    rad = 0.35 if abs(sy - dy) < 0.05 else 0.0
    ax.annotate('', xy=(dx, dy), xytext=(sx, sy),
                arrowprops=dict(arrowstyle='-|>', color='#555555',
                                lw=1.5, mutation_scale=14,
                                connectionstyle=f'arc3,rad={rad}'))

for idx, (name, x, y) in _dag_nodes.items():
    if idx < N_OBSERVABLE:
        color = CLASS_COLORS[idx]
        ax.scatter([x], [y], s=2400, color=color, zorder=3,
                   edgecolors='black', linewidths=1.0)
        ax.text(x, y, name.replace('_', '\n'), ha='center', va='center',
                color='white', fontsize=7.5, fontweight='bold', zorder=4)
    else:
        lc = LATENT_COLORS[idx - N_OBSERVABLE]
        ax.scatter([x], [y], s=2400, color=lc, zorder=3, marker='D',
                   edgecolors='black', linewidths=1.0)
        ax.text(x, y, name.replace('_', '\n'), ha='center', va='center',
                color='white', fontsize=7, fontweight='bold', zorder=4)

ax.text(0.25, 0.04, '● observable concept', color='black', fontsize=8, transform=ax.transAxes)
ax.text(0.58, 0.04, '◆ latent root cause',  color='black', fontsize=8, transform=ax.transAxes)
ax.set_title('Aircraft Damage Causal DAG', fontsize=12, pad=10)

plt.tight_layout()
plt.savefig('eval_plots/dag_structure.png', dpi=130, bbox_inches='tight')
plt.close(fig)
print('    → eval_plots/dag_structure.png')


# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  FINAL EVALUATION SUMMARY')
print(SEP)
print(f'\n  Macro AUC-ROC     : {macro_auc:.3f}')
print(f'  Mean specificity  : {mean_specificity:.3f}')
print(f'  {mic_interp_str}')
print(f'  {tic_interp_str}')
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
    'concept_activation': {
        'specificity_per_concept': spec_per_concept,
        'mean_specificity':        mean_specificity,
        'activation_table': {
            CLASS_NAMES[i]: {OBS_NAMES[j]: float(table[i, j])
                             for j in range(_N_OBS)}
            for i in range(Z1_DIM)
        },
    },
    'disentanglement': {
        'mic_overall':       overall_mic,
        'mic_per_class':     mic_per_class,
        'mic_interpretation': mic_interp_str,
        'tic_overall':       overall_tic,
        'tic_per_concept':   tic_per_concept,
        'tic_interpretation': tic_interp_str,
        'mi_matrix': {
            CLASS_NAMES[i]: {OBS_NAMES[j]: float(mi_matrix[i, j]) for j in range(_N_OBS)}
            for i in range(Z1_DIM)
        },
    },
}

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, 'w') as f:
    json.dump(report, f, indent=2)
print(f'Full report saved → {args.out}')
