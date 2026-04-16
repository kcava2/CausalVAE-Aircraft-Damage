"""
Test-set evaluation for Aircraft Damage CausalVAE.

Computes:
  Severity regression   — MAE, within-1 accuracy, exact accuracy
  Damage type detection — F1, precision, recall per class + macro
  Reconstruction        — MSE

Run from repo root:
    python evaluate_aircraft.py --checkpoint checkpoints/aircraft_best.pt
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from codebase.models.mask_vae_aircraft import CausalVAE
from dataset.aircraft_damage import get_dataloader, CLASS_NAMES, N_CONCEPTS, SCALE
from models.classifier_head import MultiLabelHead, SeverityHead, compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='checkpoints/aircraft_best.pt')
parser.add_argument('--data_root',  default='./causal_data/aircraft damage')
parser.add_argument('--split',      default='test',
                    help='Dataset split to evaluate: test, valid, or train')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--out',        default='./eval_results.json',
                    help='Where to save the JSON report')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device     : {device}')
print(f'Checkpoint : {args.checkpoint}')
print(f'Split      : {args.split}')

# ── Load model ────────────────────────────────────────────────────────────────
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
cfg  = ckpt.get('config', {'z_dim': 32, 'z1_dim': 4, 'z2_dim': 8})
Z_DIM, Z1_DIM, Z2_DIM = cfg['z_dim'], cfg['z1_dim'], cfg['z2_dim']

lvae = CausalVAE(
    z_dim=Z_DIM, z1_dim=Z1_DIM, z2_dim=Z2_DIM,
    inference=True, scale=SCALE, initial=False,
).to(device)
lvae.load_state_dict(ckpt['lvae'])
lvae.eval()

clf = MultiLabelHead(z_dim=Z_DIM, n_classes=N_CONCEPTS).to(device)
clf.load_state_dict(ckpt['clf'])
clf.eval()

sev_head = SeverityHead(z_dim=Z_DIM).to(device)
sev_head.load_state_dict(ckpt['sev_head'])
sev_head.eval()

# ── Data ──────────────────────────────────────────────────────────────────────
loader = get_dataloader(args.data_root, args.split, args.batch_size, num_workers=0)
print(f'Images     : {len(loader.dataset)}')

# ── Evaluation loop ───────────────────────────────────────────────────────────
all_logits   = []
all_labels   = []
all_sev_pred = []
all_sev_true = []
total_rec    = 0.0
n_batches    = 0

with torch.no_grad():
    for imgs, labels, sev_counts in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        sev_counts   = sev_counts.to(device)

        _, _, rec, _, z_dag = lvae.negative_elbo_bound(imgs, labels, sample=False)

        z_flat   = z_dag.reshape(imgs.size(0), -1)
        logits   = clf(z_flat)
        sev_pred = sev_head(z_flat)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_sev_pred.append(sev_pred.cpu())
        all_sev_true.append(sev_counts.cpu())
        total_rec += rec.item()
        n_batches += 1

logits_all   = torch.cat(all_logits)
labels_all   = torch.cat(all_labels)
sev_pred_all = torch.cat(all_sev_pred).float().numpy()
sev_true_all = torch.cat(all_sev_true).float().numpy()

# ── Severity metrics ──────────────────────────────────────────────────────────
sev_mae      = float(np.mean(np.abs(sev_pred_all - sev_true_all)))
sev_exact    = float(np.mean(np.round(sev_pred_all) == np.round(sev_true_all)))
sev_within1  = float(np.mean(np.abs(sev_pred_all - sev_true_all) <= 1.0))
sev_rmse     = float(np.sqrt(np.mean((sev_pred_all - sev_true_all) ** 2)))

# ── Classification metrics ────────────────────────────────────────────────────
metrics = compute_metrics(
    logits_all, labels_all,
    class_names=CLASS_NAMES,
    sev_pred=torch.tensor(sev_pred_all),
    sev_target=torch.tensor(sev_true_all),
)

avg_rec = total_rec / max(n_batches, 1)

# ── Print summary ─────────────────────────────────────────────────────────────
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
for name in CLASS_NAMES:
    f1   = metrics.get(f'f1_{name}',        0.0)
    prec = metrics.get(f'precision_{name}', 0.0)
    rec  = metrics.get(f'recall_{name}',    0.0)
    print(f'    {name:<12}  {f1:6.3f}  {prec:10.3f}  {rec:8.3f}')
print(f'    {"─"*12}  {"─"*6}  {"─"*10}  {"─"*8}')
print(f'    {"MACRO":<12}  {metrics["f1_macro"]:6.3f}  '
      f'{metrics["precision_macro"]:10.3f}  {metrics["recall_macro"]:8.3f}')

print(f'\n  RECONSTRUCTION')
print(f'    Avg MSE         : {avg_rec:.5f}')

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
            for name in CLASS_NAMES
        },
    },
    'reconstruction': {'avg_mse': avg_rec},
}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, 'w') as f:
    json.dump(report, f, indent=2)
print(f'Full report saved → {args.out}')
