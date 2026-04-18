"""
Reliability test for Aircraft Damage CausalVAE.

Loads the best checkpoint, runs evaluate_reliability() on the test split,
prints a formatted report, saves a JSON file, and saves a summary plot.

Run from repo root:
    python reliability_test.py
    python reliability_test.py --checkpoint checkpoints/aircraft_best.pt --split test
"""

import argparse

import torch

from codebase.models.mask_vae_aircraft import CausalVAE
from dataset.aircraft_damage import get_dataloader, SCALE, N_OBSERVABLE
from models.classifier_head import MultiLabelHead, SeverityHead
from reliability_metrics import evaluate_reliability, print_reliability_report, save_reliability_outputs

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='checkpoints/aircraft_best.pt')
parser.add_argument('--data_root',  default='./causal_data/aircraft damage')
parser.add_argument('--split',      default='test')
parser.add_argument('--batch_size', type=int,   default=32)
parser.add_argument('--w_I',        type=float, default=1.0,
                    help='False-positive severity weight')
parser.add_argument('--w_II',       type=float, default=1.5,
                    help='False-negative severity weight')
parser.add_argument('--epsilon',    type=float, default=1.0,
                    help='Regression failure threshold (squared error)')
parser.add_argument('--alpha',      type=float, default=0.95,
                    help='CVaR tail probability for DepS')
parser.add_argument('--sigma',      type=float, default=0.1,
                    help='Gaussian noise std in pixel space for AS')
parser.add_argument('--out_dir',    default='eval_plots',
                    help='Directory for output plot(s)')
parser.add_argument('--out_json',   default='reliability_results.json',
                    help='Path for JSON results file')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device     : {device}')
print(f'Checkpoint : {args.checkpoint}')
print(f'Split      : {args.split}')

# ── Load checkpoint ───────────────────────────────────────────────────────────
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

# ── Dataloader ────────────────────────────────────────────────────────────────
loader = get_dataloader(args.data_root, args.split, args.batch_size, num_workers=0)
print(f'Images     : {len(loader.dataset)}')

# ── Evaluate ──────────────────────────────────────────────────────────────────
results = evaluate_reliability(
    lvae, clf, sev_head, loader, device,
    w_I=args.w_I,
    w_II=args.w_II,
    epsilon=args.epsilon,
    alpha=args.alpha,
    sigma=args.sigma,
)

print_reliability_report(results)
save_reliability_outputs(results, out_dir=args.out_dir, out_json=args.out_json)
