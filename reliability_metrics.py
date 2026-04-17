"""
Reliability Metrics for Aircraft Damage CausalVAE.

Three metrics computed on top of existing model outputs without modifying
the core model or training pipeline:

  P_s   — Probability of Success: 1 - weighted failure rate
  DepS  — Dependability Score: 1 - CVaR of confidence at failure events
  AS    — Availability Score: P_s on sensor-noise-perturbed inputs

All metrics handle both supervised heads jointly:
  - MultiLabelHead: multi-label binary classifier, 4 damage concepts
  - SeverityHead:   damage instance count regressor (scalar)
"""

import numpy as np
import torch
import torch.nn.functional as F

# ImageNet normalisation constants used by the aircraft dataset transforms.
_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMG_STD  = torch.tensor([0.229, 0.224, 0.225])


# ── Pure computation helpers (no model required) ──────────────────────────────

def _clf_failure_weights(
    logits: np.ndarray,
    labels: np.ndarray,
    w_I: float,
    w_II: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-sample failure indicator and weight for classification.

    For each sample n across K=4 classes:
      - FP (Type I):  pred=1, label=0  → weight w_I
      - FN (Type II): pred=0, label=1  → weight w_II
      - failure_n = 1 if any class is misclassified
      - w(tau_n)  = weighted mean over misclassified classes

    Returns:
      failures (N,) float  — 1 if any class wrong, else 0
      weights  (N,) float  — per-sample effective error severity weight
    """
    # sigmoid threshold to get binary predictions
    probs = 1.0 / (1.0 + np.exp(-logits))          # (N, K)
    preds = (probs >= threshold).astype(float)

    fp = (preds == 1) & (labels == 0)               # (N, K) false positives
    fn = (preds == 0) & (labels == 1)               # (N, K) false negatives

    n_fp = fp.sum(axis=1).astype(float)             # (N,)
    n_fn = fn.sum(axis=1).astype(float)             # (N,)
    total_errors = n_fp + n_fn                       # (N,)

    failures = (total_errors > 0).astype(float)

    # weighted average of error types across misclassified classes;
    # safe_total avoids division by zero — np.where evaluates both branches
    safe_total = np.where(total_errors > 0, total_errors, 1.0)
    weights = np.where(
        total_errors > 0,
        (n_fp * w_I + n_fn * w_II) / safe_total,
        0.0,
    )
    return failures, weights


def _reg_failure_weights(
    sev_pred: np.ndarray,
    sev_true: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-sample failure indicator and weight for regression.

    failure_n = 1 if squared error exceeds threshold epsilon.
    Regression failures carry uniform weight 1.0 (no type distinction).

    Returns:
      failures (N,) float
      weights  (N,) float  — 1.0 at failures, 0.0 otherwise
    """
    sq_error = (sev_pred - sev_true) ** 2           # (N,)
    failures = (sq_error > epsilon).astype(float)
    weights  = failures.copy()                       # uniform weight = 1.0
    return failures, weights


def _combine_failures(
    clf_f: np.ndarray,
    clf_w: np.ndarray,
    reg_f: np.ndarray,
    reg_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Union of classification and regression failures.
    Combined weight = max(clf_weight, reg_weight) per sample.
    """
    combined_f = ((clf_f + reg_f) > 0).astype(float)
    combined_w = np.maximum(clf_w, reg_w)
    return combined_f, combined_w


def _p_failure(failures: np.ndarray, weights: np.ndarray) -> float:
    """
    P(F|D) = (1/N) * sum_n [ failure_n * w(tau_n) ]
    """
    return float(np.mean(failures * weights))


def _compute_deps(
    failures: np.ndarray,
    confidences: np.ndarray,
    alpha: float,
) -> dict:
    """
    Compute Dependability Score from per-sample failure flags and
    confidence scores.

    Collect F = { c_n | failure_n = 1 }
    VaR  = alpha-quantile of F
    CVaR = mean of F values >= VaR (Conditional Value-at-Risk tail)
    DepS = 1 - CVaR

    If no failures, returns DepS = 1.0 (perfect dependability).
    """
    failure_confs = confidences[failures == 1]

    if len(failure_confs) == 0:
        return {'VaR': float('nan'), 'CVaR': float('nan'), 'DepS': 1.0}

    var  = float(np.quantile(failure_confs, alpha))
    tail = failure_confs[failure_confs >= var]
    cvar = float(np.mean(tail))
    deps = 1.0 - cvar
    return {'VaR': var, 'CVaR': cvar, 'DepS': deps}


# ── Model inference helpers ───────────────────────────────────────────────────

def _perturb_images(
    imgs: torch.Tensor,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Simulate sensor/acquisition noise in raw pixel space [0, 1].

    Steps:
      1. Inverse-normalise from ImageNet stats → pixel values in [0, 1]
      2. Add Gaussian noise with std=sigma (sigma=0.1 ≈ 10% intensity noise)
      3. Clamp to [0, 1] to keep physically valid pixel values
      4. Re-normalise for model consumption

    This mimics real-world noise that appears before any image preprocessing.
    """
    mean = _IMG_MEAN.to(device).view(1, 3, 1, 1)
    std  = _IMG_STD.to(device).view(1, 3, 1, 1)

    imgs_raw   = imgs * std + mean                              # → [0, 1]
    imgs_noisy = (imgs_raw + torch.randn_like(imgs_raw) * sigma).clamp(0.0, 1.0)
    imgs_tilde = (imgs_noisy - mean) / std                      # re-normalise
    return imgs_tilde


@torch.no_grad()
def _run_inference(lvae, clf, sev_head, dataloader, device):
    """
    Run a full pass over the dataloader and collect raw outputs.

    Returns dict with numpy arrays:
      logits     (N, 4)  — raw classification logits
      sev_pred   (N,)    — raw regression predictions
      labels     (N, 4)  — ground-truth binary concept flags
      sev_true   (N,)    — ground-truth damage counts
    """
    all_logits, all_sev_pred, all_labels, all_sev_true = [], [], [], []

    for imgs, labels, sev_counts in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        sev_counts   = sev_counts.to(device)

        _, _, _, _, z_dag = lvae.negative_elbo_bound(imgs, labels, sample=False)
        z_flat   = z_dag.reshape(imgs.size(0), -1)
        logits   = clf(z_flat)
        sev_pred = sev_head(z_flat)

        all_logits.append(logits.cpu())
        all_sev_pred.append(sev_pred.cpu())
        all_labels.append(labels.cpu())
        all_sev_true.append(sev_counts.cpu())

    return {
        'logits':   torch.cat(all_logits).numpy(),
        'sev_pred': torch.cat(all_sev_pred).float().numpy(),
        'labels':   torch.cat(all_labels).numpy(),
        'sev_true': torch.cat(all_sev_true).float().numpy(),
    }


@torch.no_grad()
def _run_inference_perturbed(lvae, clf, sev_head, dataloader, device, sigma):
    """
    Same as _run_inference but with Gaussian sensor noise applied to images
    in raw pixel space before passing through the model.
    """
    all_logits, all_sev_pred, all_labels, all_sev_true = [], [], [], []

    for imgs, labels, sev_counts in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        sev_counts   = sev_counts.to(device)

        # apply sensor-noise perturbation in pixel space, then re-normalise
        imgs_tilde = _perturb_images(imgs, sigma, device)

        _, _, _, _, z_dag = lvae.negative_elbo_bound(imgs_tilde, labels, sample=False)
        z_flat   = z_dag.reshape(imgs_tilde.size(0), -1)
        logits   = clf(z_flat)
        sev_pred = sev_head(z_flat)

        all_logits.append(logits.cpu())
        all_sev_pred.append(sev_pred.cpu())
        all_labels.append(labels.cpu())
        all_sev_true.append(sev_counts.cpu())

    return {
        'logits':   torch.cat(all_logits).numpy(),
        'sev_pred': torch.cat(all_sev_pred).float().numpy(),
        'labels':   torch.cat(all_labels).numpy(),
        'sev_true': torch.cat(all_sev_true).float().numpy(),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_reliability(
    lvae,
    clf,
    sev_head,
    dataloader,
    device,
    *,
    w_I: float       = 1.0,
    w_II: float      = 1.5,
    epsilon: float   = 1.0,
    alpha: float     = 0.95,
    sigma: float     = 0.1,
    threshold: float = 0.5,
) -> dict:
    """
    Compute all three reliability metrics on the given dataloader.

    Parameters
    ----------
    lvae, clf, sev_head : loaded model objects (eval mode)
    dataloader          : yields (imgs, labels, sev_counts) batches
    device              : torch device
    w_I                 : false-positive severity weight (Type I error)
    w_II                : false-negative severity weight (Type II error)
    epsilon             : regression failure threshold (squared error)
    alpha               : CVaR tail probability for DepS (default 0.95)
    sigma               : Gaussian noise std in pixel space [0,1] for AS
    threshold           : sigmoid threshold for binary prediction (default 0.5)

    Returns
    -------
    dict with keys:
      P_F_D_clf, P_F_D_reg, P_F_D_combined   — failure probabilities (clean)
      P_s_clf, P_s_reg, P_s_combined          — success probabilities
      VaR_clf,  CVaR_clf,  DepS_clf
      VaR_reg,  CVaR_reg,  DepS_reg
      DepS_combined
      P_F_D_tilde_clf, P_F_D_tilde_reg, P_F_D_tilde_combined — perturbed failure probs
      AS_clf, AS_reg, AS_combined             — availability per head + combined
      n_samples, n_failures_clf, n_failures_reg, n_failures_combined
      w_I, w_II, epsilon, alpha, sigma
    """
    # ── Step 1: inference on clean data ──────────────────────────────────────
    out = _run_inference(lvae, clf, sev_head, dataloader, device)
    logits, sev_pred = out['logits'], out['sev_pred']
    labels, sev_true = out['labels'], out['sev_true']
    N = len(logits)

    # ── Step 2: Probability of Failure (P_s) ─────────────────────────────────
    clf_f, clf_w = _clf_failure_weights(logits, labels, w_I, w_II, threshold)
    reg_f, reg_w = _reg_failure_weights(sev_pred, sev_true, epsilon)
    comb_f, comb_w = _combine_failures(clf_f, clf_w, reg_f, reg_w)

    p_f_clf  = _p_failure(clf_f,  clf_w)
    p_f_reg  = _p_failure(reg_f,  reg_w)
    p_f_comb = _p_failure(comb_f, comb_w)

    # ── Step 3: Dependability Score (DepS) ───────────────────────────────────
    # Classification confidence: max(softmax(logits)) at failure events
    clf_conf = F.softmax(torch.tensor(logits), dim=1).max(dim=1).values.numpy()

    # Regression confidence: inverted normalised absolute error at failure events
    abs_err  = np.abs(sev_pred - sev_true)
    reg_fail_mask = reg_f == 1
    max_err  = abs_err[reg_fail_mask].max() if reg_fail_mask.any() else 1.0
    max_err  = max_err if max_err > 0 else 1.0                # avoid division by zero
    reg_conf = 1.0 - (abs_err / max_err)

    # Combined confidence: use clf conf for clf-only, reg conf for reg-only,
    # max of both when both heads fail on the same sample.
    combined_conf = np.where(
        (clf_f == 1) & (reg_f == 1), np.maximum(clf_conf, reg_conf),
        np.where(clf_f == 1, clf_conf, reg_conf),
    )

    deps_clf  = _compute_deps(clf_f,  clf_conf,     alpha)
    deps_reg  = _compute_deps(reg_f,  reg_conf,     alpha)
    deps_comb = _compute_deps(comb_f, combined_conf, alpha)

    # ── Step 4: Availability Score (AS) — perturbed data ─────────────────────
    out_tilde = _run_inference_perturbed(lvae, clf, sev_head, dataloader, device, sigma)
    clf_f_t, clf_w_t = _clf_failure_weights(
        out_tilde['logits'], out_tilde['labels'], w_I, w_II, threshold
    )
    reg_f_t, reg_w_t = _reg_failure_weights(
        out_tilde['sev_pred'], out_tilde['sev_true'], epsilon
    )
    comb_f_t, comb_w_t = _combine_failures(clf_f_t, clf_w_t, reg_f_t, reg_w_t)
    p_f_tilde_clf  = _p_failure(clf_f_t,  clf_w_t)
    p_f_tilde_reg  = _p_failure(reg_f_t,  reg_w_t)
    p_f_tilde_comb = _p_failure(comb_f_t, comb_w_t)

    return {
        # Probability of Failure
        'P_F_D_clf':               p_f_clf,
        'P_F_D_reg':               p_f_reg,
        'P_F_D_combined':          p_f_comb,
        'P_s_clf':                 1.0 - p_f_clf,
        'P_s_reg':                 1.0 - p_f_reg,
        'P_s_combined':            1.0 - p_f_comb,
        # Dependability Score
        'VaR_clf':                 deps_clf['VaR'],
        'CVaR_clf':                deps_clf['CVaR'],
        'DepS_clf':                deps_clf['DepS'],
        'VaR_reg':                 deps_reg['VaR'],
        'CVaR_reg':                deps_reg['CVaR'],
        'DepS_reg':                deps_reg['DepS'],
        'DepS_combined':           deps_comb['DepS'],
        # Availability Score
        'P_F_D_tilde_clf':         p_f_tilde_clf,
        'P_F_D_tilde_reg':         p_f_tilde_reg,
        'P_F_D_tilde_combined':    p_f_tilde_comb,
        'AS_clf':                  1.0 - p_f_tilde_clf,
        'AS_reg':                  1.0 - p_f_tilde_reg,
        'AS_combined':             1.0 - p_f_tilde_comb,
        # Diagnostics
        'n_samples':           N,
        'n_failures_clf':      int(clf_f.sum()),
        'n_failures_reg':      int(reg_f.sum()),
        'n_failures_combined': int(comb_f.sum()),
        'w_I':                 w_I,
        'w_II':                w_II,
        'epsilon':             epsilon,
        'alpha':               alpha,
        'sigma':               sigma,
    }


def print_reliability_report(results: dict) -> None:
    """Print a formatted summary of all reliability metrics."""
    SEP = '─' * 56
    N   = results['n_samples']

    print(f'\n{SEP}')
    print('  RELIABILITY METRICS  (CausalVAE Aircraft Damage)')
    print(SEP)
    print(f'  Samples : {N}  |  w_I={results["w_I"]}  w_II={results["w_II"]}  '
          f'ε={results["epsilon"]}  α={results["alpha"]}  σ={results["sigma"]}')

    print(f'\n  PROBABILITY OF SUCCESS  P_s = 1 - P(F|D)')
    print(f'    {"Head":<14}  {"P(F|D)":>8}  {"P_s":>8}  {"# Failures":>12}')
    print(f'    {"─"*14}  {"─"*8}  {"─"*8}  {"─"*12}')
    print(f'    {"Classification":<14}  {results["P_F_D_clf"]:8.4f}  '
          f'{results["P_s_clf"]:8.4f}  {results["n_failures_clf"]:12d}')
    print(f'    {"Regression":<14}  {results["P_F_D_reg"]:8.4f}  '
          f'{results["P_s_reg"]:8.4f}  {results["n_failures_reg"]:12d}')
    print(f'    {"Combined":<14}  {results["P_F_D_combined"]:8.4f}  '
          f'{results["P_s_combined"]:8.4f}  {results["n_failures_combined"]:12d}')

    print(f'\n  DEPENDABILITY SCORE  DepS = 1 - CVaR(confidence | failure)')
    print(f'    {"Head":<14}  {"VaR":>8}  {"CVaR":>8}  {"DepS":>8}')
    print(f'    {"─"*14}  {"─"*8}  {"─"*8}  {"─"*8}')
    for head, key_var, key_cvar, key_deps in [
        ('Classification', 'VaR_clf', 'CVaR_clf', 'DepS_clf'),
        ('Regression',     'VaR_reg', 'CVaR_reg', 'DepS_reg'),
        ('Combined',       None,      None,        'DepS_combined'),
    ]:
        var_s  = f'{results[key_var]:.4f}'  if key_var  and results[key_var]  == results[key_var]  else 'N/A'
        cvar_s = f'{results[key_cvar]:.4f}' if key_cvar and results[key_cvar] == results[key_cvar] else 'N/A'
        print(f'    {head:<14}  {var_s:>8}  {cvar_s:>8}  {results[key_deps]:8.4f}')

    print(f'\n  AVAILABILITY SCORE  AS = 1 - P(F|D̃)  (σ={results["sigma"]} sensor noise)')
    print(f'    {"Head":<14}  {"P(F|D̃)":>8}  {"AS":>8}  {"ΔP_F (noise)":>14}')
    print(f'    {"─"*14}  {"─"*8}  {"─"*8}  {"─"*14}')
    for head, key_tilde, key_clean, key_as in [
        ('Classification', 'P_F_D_tilde_clf',      'P_F_D_clf',      'AS_clf'),
        ('Regression',     'P_F_D_tilde_reg',      'P_F_D_reg',      'AS_reg'),
        ('Combined',       'P_F_D_tilde_combined', 'P_F_D_combined', 'AS_combined'),
    ]:
        delta = results[key_tilde] - results[key_clean]
        print(f'    {head:<14}  {results[key_tilde]:8.4f}  {results[key_as]:8.4f}  {delta:+14.4f}')

    print(f'\n{SEP}\n')
