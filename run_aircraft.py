"""
Training script for CausalVAE on the Aircraft Damage dataset.
4 supervised concepts: crack, dent, paint_off, scratch.
Primary prediction: damage instance count (severity regression).

Run from repo root:
    python run_aircraft.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from codebase import utils as ut
from codebase.models.mask_vae_aircraft import CausalVAE
from dataset.aircraft_damage import (
    get_dataloader, CLASS_NAMES, N_OBSERVABLE, SCALE
)
from dataset.aircraft_dag import A_INIT
from models.classifier_head import (
    MultiLabelHead, SeverityHead, ConceptHeads,
    multilabel_loss, severity_loss, compute_metrics,
)
from utils import _h_A

# ── Config ────────────────────────────────────────────────────────────────────
Z1_DIM     = 7                    # concepts: 4 observable + 3 latent root causes
Z2_DIM     = 4                    # features per concept
Z_DIM      = Z1_DIM * Z2_DIM     # = 28
EPOCHS     = 201
BATCH_SIZE = 64
LR         = 1e-4
SEV_WEIGHT = 2.0    # severity is the primary supervised signal
CLF_WEIGHT = 1.0    # concept labels keep DAG structure meaningful
DATA_ROOT  = './causal_data/aircraft damage'
SAVE_DIR   = './checkpoints'
FIG_DIR    = './figs_aircraft'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ── Data ──────────────────────────────────────────────────────────────────────
train_loader = get_dataloader(DATA_ROOT, 'train', BATCH_SIZE, num_workers=0)
val_loader   = get_dataloader(DATA_ROOT, 'valid', BATCH_SIZE, num_workers=0)

# Class balance for pos_weight (only for N_OBSERVABLE observable concepts)
print('Computing class statistics…')
pos_counts = np.zeros(N_OBSERVABLE, dtype=float)
n_total    = 0
for _, labels, _ in train_loader:
    pos_counts += labels[:, :N_OBSERVABLE].sum(0).numpy()
    n_total    += labels.size(0)
neg_counts = n_total - pos_counts
pos_weight = torch.tensor(
    neg_counts / np.maximum(pos_counts, 1.0), dtype=torch.float32
).to(device)
print(f'  pos_weight: {pos_weight.cpu().numpy().round(2)}')
print(f'  pos_rates:  {(pos_counts/n_total).round(3)}')

# ── Model ─────────────────────────────────────────────────────────────────────
lvae = CausalVAE(
    name='aircraft_causalvae',
    z_dim=Z_DIM,
    z1_dim=Z1_DIM,
    z2_dim=Z2_DIM,
    inference=False,
    scale=SCALE,
    initial=False,
).to(device)

with torch.no_grad():
    lvae.dag.A.copy_(A_INIT.to(device))
print('DAG initialised:\n', lvae.dag.A.detach().cpu().numpy())

clf           = MultiLabelHead(z_dim=Z_DIM, n_classes=N_OBSERVABLE).to(device)
sev_head      = SeverityHead(z_dim=Z_DIM).to(device)
concept_heads = ConceptHeads(z2_dim=Z2_DIM, n_observable=N_OBSERVABLE).to(device)

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

# ── Optimiser ─────────────────────────────────────────────────────────────────
enc_params   = list(lvae.enc.parameters()) + list(lvae.enc_proj.parameters())
dec_params   = list(lvae.dec.parameters())
other_params = (
    list(lvae.dag.parameters())    +
    list(lvae.attn.parameters())   +
    list(lvae.mask_z.parameters()) +
    list(lvae.mask_u.parameters())
)

optimizer = torch.optim.Adam([
    {'params': enc_params,                   'lr': LR},
    {'params': dec_params,                   'lr': 5e-4},
    {'params': other_params,                 'lr': LR},
    {'params': clf.parameters(),             'lr': LR},
    {'params': sev_head.parameters(),        'lr': LR},
    {'params': concept_heads.parameters(),   'lr': LR},
], betas=(0.9, 0.999))

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
)

# ── KL annealing ──────────────────────────────────────────────────────────────
def kl_weight(epoch: int) -> float:
    return min(1.0, epoch / 50.0) * 0.25

# ── Helpers ───────────────────────────────────────────────────────────────────
def total_correlation_loss(z_dag: torch.Tensor) -> torch.Tensor:
    act   = torch.sigmoid(z_dag.mean(dim=-1))           # (B, Z1_DIM)
    act_c = act - act.mean(dim=0, keepdim=True)
    cov   = (act_c.T @ act_c) / act.size(0)             # (Z1_DIM, Z1_DIM)
    eye   = torch.eye(cov.size(0), device=cov.device)
    return ((cov * (1 - eye)) ** 2).sum()


def denorm(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(t.device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(t.device)
    return torch.clamp(t * std + mean, 0, 1)


def save_recon_samples(imgs, recons, epoch, n=8):
    imgs   = imgs[:n].detach().cpu()
    recons = recons[:n].detach().cpu()
    grid   = torch.cat([denorm(imgs), denorm(recons)], dim=0)
    save_image(grid, os.path.join(FIG_DIR, f'recon_epoch{epoch:04d}.png'), nrow=n)


def validate(model, clf_model, sev_model, loader):
    model.eval(); clf_model.eval(); sev_model.eval(); concept_heads.eval()
    all_logits, all_targets = [], []
    all_sev_pred, all_sev_tgt = [], []
    total_rec = total_kl = 0.0
    n_batches = 0
    with torch.no_grad():
        for imgs, labels, sev_counts in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            sev_counts   = sev_counts.to(device)

            _, kl, rec, _, z_dag = model.negative_elbo_bound(
                imgs, labels, sample=False
            )
            z_flat  = z_dag.reshape(imgs.size(0), -1)
            logits  = clf_model(z_flat)
            sev_pred = sev_model(z_flat)

            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())
            all_sev_pred.append(sev_pred.cpu())
            all_sev_tgt.append(sev_counts.cpu())
            total_rec += rec.item()
            total_kl  += kl.item()
            n_batches += 1

    logits_all   = torch.cat(all_logits,    dim=0)
    targets_all  = torch.cat(all_targets,   dim=0)
    sev_pred_all = torch.cat(all_sev_pred,  dim=0)
    sev_tgt_all  = torch.cat(all_sev_tgt,   dim=0)

    metrics = compute_metrics(
        logits_all, targets_all[:, :N_OBSERVABLE],
        class_names=CLASS_NAMES[:N_OBSERVABLE],
        sev_pred=sev_pred_all,
        sev_target=sev_tgt_all,
    )
    metrics['rec'] = total_rec / max(n_batches, 1)
    metrics['kl']  = total_kl  / max(n_batches, 1)
    return metrics


def save_checkpoint(epoch, val_sev_mae, tag='best'):
    path = os.path.join(SAVE_DIR, f'aircraft_{tag}.pt')
    torch.save({
        'lvae':          lvae.state_dict(),
        'clf':           clf.state_dict(),
        'sev_head':      sev_head.state_dict(),
        'concept_heads': concept_heads.state_dict(),
        'epoch':         epoch,
        'val_sev_mae':   val_sev_mae,
        'config': {'z_dim': Z_DIM, 'z1_dim': Z1_DIM, 'z2_dim': Z2_DIM, 'n_observable': N_OBSERVABLE},
    }, path)
    print(f'  Saved checkpoint → {path}')


def plot_history(history, save_path):
    epochs = range(1, len(history['loss']) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Aircraft CausalVAE Training', fontsize=13)

    specs = [
        (axes[0, 0], 'loss',    'Total Loss',          'tab:blue'),
        (axes[0, 1], 'kl',      'KL Loss',             'tab:orange'),
        (axes[0, 2], 'rec',     'Reconstruction Loss', 'tab:green'),
        (axes[1, 0], 'clf',     'Concept BCE Loss',    'tab:red'),
        (axes[1, 1], 'sev',     'Severity Loss',       'tab:purple'),
        (axes[1, 2], 'sev_mae', 'Val Severity MAE',    'tab:brown'),
    ]
    for ax, key, title, color in specs:
        if key in history and history[key]:
            ax.plot(range(1, len(history[key]) + 1),
                    history[key], color=color)
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)


# ── Training loop ─────────────────────────────────────────────────────────────
history = {
    'loss': [], 'kl': [], 'rec': [], 'clf': [], 'sev': [], 'sev_mae': [],
}
best_val_sev_mae = float('inf')

for epoch in range(EPOCHS):
    lvae.train(); clf.train(); sev_head.train(); concept_heads.train()

    kl_w       = kl_weight(epoch)
    total_loss = total_kl = total_rec = total_clf = total_sev = 0.0
    last_recon = last_imgs = None

    for imgs, labels, sev_counts in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        sev_counts   = sev_counts.to(device)
        optimizer.zero_grad()

        nelbo, kl, rec, recon, z_dag = lvae.negative_elbo_bound(
            imgs, labels, sample=False
        )

        # DAG acyclicity penalty (NOTEARS)
        h_a       = _h_A(lvae.dag.A, lvae.dag.A.size(0))
        dag_penalty = 3 * h_a + 0.5 * h_a * h_a

        z_flat   = z_dag.reshape(imgs.size(0), -1)
        logits   = clf(z_flat)
        sev_pred = sev_head(z_flat)

        clf_loss     = multilabel_loss(logits, labels[:, :N_OBSERVABLE], pos_weight=pos_weight)
        sev_l        = severity_loss(sev_pred, sev_counts)
        concept_logits = concept_heads(z_dag)
        concept_loss = multilabel_loss(concept_logits, labels[:, :N_OBSERVABLE], pos_weight=pos_weight)
        tc_loss      = total_correlation_loss(z_dag)

        mask_l = nelbo - rec - kl
        loss = (rec
                + kl_w * kl
                + 0.2 * mask_l
                + dag_penalty
                + SEV_WEIGHT * sev_l
                + CLF_WEIGHT * clf_loss
                + 0.5 * concept_loss
                + 0.1 * tc_loss)

        loss.backward()
        nn.utils.clip_grad_norm_(
            list(lvae.parameters()) + list(clf.parameters()) +
            list(sev_head.parameters()) + list(concept_heads.parameters()),
            max_norm=5.0,
        )
        optimizer.step()

        total_loss += loss.item()
        total_kl   += kl.item()
        total_rec  += rec.item()
        total_clf  += clf_loss.item()
        total_sev  += sev_l.item()
        last_recon, last_imgs = recon, imgs

    n_batches = len(train_loader)
    avg_loss  = total_loss / n_batches
    avg_kl    = total_kl   / n_batches
    avg_rec   = total_rec  / n_batches
    avg_clf   = total_clf  / n_batches
    avg_sev   = total_sev  / n_batches

    val_metrics  = validate(lvae, clf, sev_head, val_loader)
    val_sev_mae  = val_metrics.get('sev_mae', float('inf'))
    val_sev_acc  = val_metrics.get('sev_acc', 0.0)
    val_f1       = val_metrics.get('f1_macro', 0.0)
    scheduler.step(val_sev_mae)

    history['loss'].append(avg_loss)
    history['kl'].append(avg_kl)
    history['rec'].append(avg_rec)
    history['clf'].append(avg_clf)
    history['sev'].append(avg_sev)
    history['sev_mae'].append(val_sev_mae)

    per_class_str = '  '.join(
        f'{n}={val_metrics.get(f"f1_{n}", 0.0):.3f}' for n in CLASS_NAMES[:N_OBSERVABLE]
    )
    cur_lr = optimizer.param_groups[0]['lr']
    print(
        f'[{epoch:03d}/{EPOCHS}] loss={avg_loss:.4f}  rec={avg_rec:.4f}  '
        f'sev={avg_sev:.4f}  clf={avg_clf:.4f}  '
        f'val_sev_mae={val_sev_mae:.3f}  val_sev_acc={val_sev_acc:.3f}  '
        f'val_f1={val_f1:.3f}  kl_w={kl_w:.3f}  lr={cur_lr:.2e}\n'
        f'         per-class: {per_class_str}'
    )

    if val_sev_mae < best_val_sev_mae:
        best_val_sev_mae = val_sev_mae
        save_checkpoint(epoch, val_sev_mae, tag='best')

    if epoch % 10 == 0:
        save_checkpoint(epoch, val_sev_mae, tag=f'epoch{epoch:04d}')
        if last_recon is not None:
            save_recon_samples(last_imgs, last_recon, epoch)

    plot_history(history, os.path.join(FIG_DIR, 'training_metrics.png'))

# ── Final checkpoint ──────────────────────────────────────────────────────────
save_checkpoint(EPOCHS - 1, best_val_sev_mae, tag='final')
print(f'\nTraining complete. Best val severity MAE: {best_val_sev_mae:.4f}')

# ── Precompute skip-channel correlations ──────────────────────────────────────
print('\nComputing skip-channel correlations for intervention…')
from codebase.models.mask_vae_aircraft import compute_skip_channel_correlations
corr_idx  = compute_skip_channel_correlations(lvae, val_loader, device, top_k=16)
corr_path = os.path.join(SAVE_DIR, 'corr_idx.pt')
torch.save(corr_idx, corr_path)
print(f'  Saved → {corr_path}')

summary = {
    'best_val_sev_mae': best_val_sev_mae,
    'final_loss':       history['loss'][-1] if history['loss'] else None,
    'final_rec':        history['rec'][-1]  if history['rec']  else None,
    'config': {
        'z_dim': Z_DIM, 'z1_dim': Z1_DIM, 'z2_dim': Z2_DIM, 'n_observable': N_OBSERVABLE,
        'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
        'sev_weight': SEV_WEIGHT, 'clf_weight': CLF_WEIGHT,
    },
}
with open(os.path.join(SAVE_DIR, 'training_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print('Summary saved.')
