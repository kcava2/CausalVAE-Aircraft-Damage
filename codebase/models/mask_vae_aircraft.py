import torch
import numpy as np
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F


class CausalVAE(nn.Module):
    def __init__(
        self,
        nn_name='mask',
        name='vae_aircraft',
        z_dim=32,
        z1_dim=4,
        z2_dim=8,
        inference=False,
        alpha=0.3,
        beta=1,
        scale=None,
        initial=True,
    ):
        super().__init__()

        self.name   = name
        self.z_dim  = z_dim       # total latent dim = z1_dim * z2_dim
        self.z1_dim = z1_dim      # number of concepts (4)
        self.z2_dim = z2_dim      # features per concept (8)
        self.channel = 3

        # scale must be provided from dataset.aircraft_damage.SCALE (shape 4×2)
        if scale is None:
            raise ValueError('Pass scale=SCALE from dataset.aircraft_damage')
        self.scale = scale

        nn_module = getattr(nns, nn_name)

        # UNet encoder: returns (feat_128, skips)
        self.enc = nn_module.UNetEncoderVAE(channel=self.channel)

        # Project 64-d mean (half of 128 after gaussian_parameters) to z_dim
        self.enc_proj = nn.Linear(64, self.z_dim)

        # UNet decoder: accepts (z_4d, skips) — used for reconstruction
        self.dec = nn_module.UNetDecoderVAE(
            in_features=self.z_dim,
            channel=self.channel,
        )

        # DAG layer over z1_dim=4 concepts
        self.dag    = nn_module.DagLayer(self.z1_dim, self.z1_dim,
                                         i=inference, initial=initial)
        self.attn   = nn_module.Attention(self.z2_dim)
        self.mask_z = nn_module.MaskLayer(self.z_dim,   concept=self.z1_dim,
                                          z2_dim=self.z2_dim)
        self.mask_u = nn_module.MaskLayer(self.z1_dim,  concept=self.z1_dim,
                                          z2_dim=1)

    # ------------------------------------------------------------------
    def negative_elbo_bound(
        self,
        x,
        label,
        mask=None,
        sample=False,
        adj=None,
        alpha=0.3,
        beta=1,
        lambdav=0.001,
    ):
        """
        Compute ELBO, KL, and MSE reconstruction loss.

        Args:
            x:       (batch, 3, 64, 64)   input images
            label:   (batch, z1_dim)       concept labels (4 binary flags)
            mask:    int or None           concept index to zero (counterfactual)
            sample:  bool                  if True use prior sample (unused in training)
            adj:     float                 value to inject when masking
            alpha:   float                 weight for encoder KL
            beta:    float                 weight for DAG KL
            lambdav: float                 variance scaling for posterior sample

        Returns:
            nelbo, kl, rec, recon_img, z_given_dag
        """
        assert label.size(1) == self.z1_dim

        dev   = x.device
        batch = x.size(0)

        # ── Encode ────────────────────────────────────────────────────
        feat, skips = self.enc.encode(x)          # feat: (batch, 128, 1, 1)
        q_m, q_v    = ut.gaussian_parameters(feat, dim=1)  # (batch, 64, 1, 1) each
        q_m = self.enc_proj(q_m.view(batch, -1))           # (batch, z_dim)
        q_m = q_m.reshape([batch, self.z1_dim, self.z2_dim])
        q_v = torch.ones(batch, self.z1_dim, self.z2_dim).to(dev)

        # ── DAG transform ─────────────────────────────────────────────
        decode_m, decode_v = self.dag.calculate_dag(
            q_m.to(dev),
            torch.ones(batch, self.z1_dim, self.z2_dim).to(dev),
        )
        decode_m = decode_m.reshape([batch, self.z1_dim, self.z2_dim])

        if not sample:
            # Concept masking (counterfactual mode)
            if mask is not None and mask < self.z1_dim - 1:
                z_mask = torch.ones(batch, self.z1_dim, self.z2_dim).to(dev) * adj
                decode_m[:, mask, :] = z_mask[:, mask, :]
                decode_v[:, mask, :] = z_mask[:, mask, :]

            m_zm = self.dag.mask_z(decode_m.to(dev)).reshape(
                [batch, self.z1_dim, self.z2_dim])
            m_zv = decode_v.reshape([batch, self.z1_dim, self.z2_dim])
            m_u  = self.dag.mask_u(label.to(dev))

            f_z     = self.mask_z.mix(m_zm).reshape(
                [batch, self.z1_dim, self.z2_dim]).to(dev)
            e_tilde = self.attn.attention(
                decode_m.reshape([batch, self.z1_dim, self.z2_dim]).to(dev),
                q_m.reshape([batch, self.z1_dim, self.z2_dim]).to(dev),
            )[0]

            if mask is not None and mask < self.z1_dim - 1:
                z_mask = torch.ones(batch, self.z1_dim, self.z2_dim).to(dev) * adj
                e_tilde[:, mask, :] = z_mask[:, mask, :]

            f_z1 = f_z + e_tilde

            # Late-concept masking
            if mask is not None and mask >= self.z1_dim - 2:
                z_mask = torch.ones(batch, self.z1_dim, self.z2_dim).to(dev) * adj
                f_z1[:, mask, :] = z_mask[:, mask, :]
                m_zv[:, mask, :] = z_mask[:, mask, :]

            g_u = self.mask_u.mix(m_u).to(dev)
            z_given_dag = f_z1 + (m_zv * lambdav).sqrt() * torch.randn_like(f_z1)

        # ── Decode ────────────────────────────────────────────────────
        z_4d  = z_given_dag.reshape([batch, self.z_dim, 1, 1])
        recon = self.dec.decode(z_4d, skips)    # (batch, 3, 64, 64)

        # ── Reconstruction losses ──────────────────────────────────────
        x_dev = x.to(dev)

        # Primary: UNet decoder (sharp, uses skips)
        rec = F.mse_loss(recon, x_dev)
        rec = (rec
               + 0.5  * F.mse_loss(F.avg_pool2d(recon, 2), F.avg_pool2d(x_dev, 2))
               + 0.25 * F.mse_loss(F.avg_pool2d(recon, 4), F.avg_pool2d(x_dev, 4)))

        # ── KL losses ─────────────────────────────────────────────────
        p_m = torch.zeros(q_m.size()).to(dev)
        p_v = torch.ones(q_m.size()).to(dev)

        cp_m, cp_v = ut.condition_prior(self.scale, label, self.z2_dim)
        cp_v = torch.ones([batch, self.z1_dim, self.z2_dim]).to(dev)
        cp_m = cp_m.to(dev)

        kl = alpha * ut.kl_normal(
            q_m.view(-1, self.z_dim).to(dev),
            q_v.view(-1, self.z_dim).to(dev),
            p_m.view(-1, self.z_dim),
            p_v.view(-1, self.z_dim),
        )
        for i in range(self.z1_dim):
            kl = kl + beta * ut.kl_normal(
                decode_m[:, i, :].to(dev),
                cp_v[:, i, :].to(dev),
                cp_m[:, i, :],
                cp_v[:, i, :],
            )
        kl = torch.mean(kl)

        mask_kl = torch.zeros(1).to(dev)
        for i in range(self.z1_dim):
            mask_kl = mask_kl + ut.kl_normal(
                f_z1[:, i, :].to(dev),
                cp_v[:, i, :].to(dev),
                cp_m[:, i, :],
                cp_v[:, i, :],
            )

        u_loss = torch.nn.MSELoss()
        mask_l = torch.mean(mask_kl) + u_loss(g_u, label.float().to(dev))
        nelbo  = rec + kl + mask_l

        return nelbo, kl, rec, recon, z_given_dag

    def loss(self, x):
        nelbo, kl, rec, _, _ = self.negative_elbo_bound(x)
        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))
        return nelbo, summaries


@torch.no_grad()
def compute_skip_channel_correlations(model, dataloader, device, top_k=32):
    """
    For each skip level and each concept k, find the top_k skip channels
    most correlated with the ground-truth concept label for k across the dataset.

    Using binary labels (not z norms) gives a direct signal: channels that are
    consistently active when concept k is present and quiet when it is absent.

    Returns:
        corr_idx: dict {level_name: tensor (z1_dim, top_k)}  (channel indices)
    """
    model.eval()
    level_names = ['s1', 's2', 's3', 's4', 'b']

    skip_acts   = {name: [] for name in level_names}
    label_acc   = []

    for imgs, labels, _ in dataloader:
        imgs = imgs.to(device)
        _, skips = model.enc.encode(imgs)

        # Ground-truth concept labels: (batch, z1_dim) float
        label_acc.append(labels.float().cpu())

        # Skip spatial mean: (batch, C) per level
        for name, s in zip(level_names, skips):
            act = s.mean(dim=(2, 3)) if s.dim() == 4 else s.view(s.size(0), -1)
            skip_acts[name].append(act.cpu())

    labels_all = torch.cat(label_acc, dim=0)   # (N, z1_dim)

    corr_idx = {}
    for name in level_names:
        acts = torch.cat(skip_acts[name], dim=0)   # (N, C)
        acts_c = acts   - acts.mean(0,   keepdim=True)
        lab_c  = labels_all - labels_all.mean(0, keepdim=True)
        acts_n = acts_c / acts_c.std(0,   keepdim=True).clamp(min=1e-8)
        lab_n  = lab_c  / lab_c.std(0,  keepdim=True).clamp(min=1e-8)

        # Pearson: (z1_dim, C) via matmul
        corr     = (lab_n.T @ acts_n) / acts_n.size(0)   # (z1_dim, C)
        k        = min(top_k, acts.size(1))
        corr_idx[name] = corr.abs().topk(k, dim=1).indices  # (z1_dim, k)

    return corr_idx
