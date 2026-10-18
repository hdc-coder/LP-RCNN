# -*- coding: utf-8 -*-
"""Hyper-parameter sweep: learning rate x batch size for LP-RCNN.

Selection uses synthetic graphs only (train BA(1000,4), validate on BA(2000,4)),
so the nine-network benchmark stays a clean test set.  Phase 1 screens the grid
and marks diverged runs; selection keeps the gate-passers and takes the Pareto
frontier on (tau up, train time down, time-to-target down).  Phase 2 retrains
that frontier at 500 epochs and three seeds.

Writes ``figures/lr_batch_sweep/``: one CSV per phase, heatmap, Pareto scatter.
"""
import os
import time
import argparse
import warnings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Utils
import Embeddings
import Models
import evaluate
from evaluate import DEVICE, L

SEED = 10
LRS = [0.01, 0.005, 0.002, 0.001, 0.0005]
BATCHES = [4, 8, 16, 32, 64]
EPOCHS = 300            # screening budget (confirmation uses 500)
VAL_EVERY = 20
THRESHOLD = 0.8

TRAIN_NET_PATH = './Networks/training/Train_1000_4.txt'

# held-out validation network: an independent BA realisation of another size,
# so no real-world benchmark takes part in the selection
VAL_NET_NAME = 'BA(2000,4)'
VAL_NET_PATH = './Networks/training/BA_2000_4.txt'
VAL_SIR_PREFIX = './SIR results/BA_2000_4/BA_2000_4_'

OUT_DIR = os.path.join('figures', 'lr_batch_sweep')
os.makedirs(OUT_DIR, exist_ok=True)
SWEEP_CSV = os.path.join(OUT_DIR, 'lr_batch_sweep.csv')
CONFIRM_CSV = os.path.join(OUT_DIR, 'lr_batch_confirm.csv')
CURVES_CSV = os.path.join(OUT_DIR, 'lr_batch_curves.csv')


# ---------------------------------------------------------------------------
# Shared data (built once, reused by every combination)
# ---------------------------------------------------------------------------

def load_shared():
    import train as train_mod
    G, label = train_mod.load_training_data()
    emb_train = Embeddings.main_ndrcnn(G, L)
    Gv = Utils.load_graph(VAL_NET_PATH)
    sirs = Utils.load_sir_list(VAL_SIR_PREFIX, n=len(evaluate.BETA_RATIOS))
    nodes_v = list(Gv.nodes())
    Xv = evaluate.to_tensor(Embeddings.main_ndrcnn(Gv, L), nodes_v, 1, L)
    return emb_train, label, [(Xv, nodes_v, sirs)]


def build_loader(emb, label, batch):
    from torch.utils.data import DataLoader, TensorDataset
    X, nodes = evaluate.to_tensor(emb, list(emb.keys()), 1, L), list(emb.keys())
    y = torch.zeros(len(nodes), 1)
    for i, node in enumerate(nodes):
        y[i, 0] = label.get(node, 0.0)
    return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=True)


def validate(model, val_data):
    taus = []
    model.eval()
    with torch.no_grad():
        for Xv, nodes_v, sirs in val_data:
            scores = model(Xv.to(DEVICE)).cpu().numpy().flatten()
            taus.append(np.mean(evaluate.kendall_taus(scores, nodes_v, sirs)))
    return float(np.mean(taus))


# ---------------------------------------------------------------------------
# Phase 1 -- screening
# ---------------------------------------------------------------------------

def train_combo(lr, batch, emb_train, label, val_data, max_epochs, val_every):
    """Train one (lr, batch) combination; return a result dict + curve."""
    Utils.setup_seed(SEED)
    loader = build_loader(emb_train, label, batch)
    model = Models.DC_RCNN_MS_DIL(L, in_channels=1).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    curve = []
    first_cross = None
    status = 'ok'
    t_start = time.perf_counter()

    for ep in range(1, max_epochs + 1):
        model.train()
        for data, tgt in loader:
            data = data.to(DEVICE)
            tgt = tgt.float().to(DEVICE)
            opt.zero_grad()
            loss = Utils.listmle_loss(model(data), tgt)
            if torch.isnan(loss):
                status = 'diverged'
                break
            loss.backward()
            opt.step()
        if status == 'diverged':
            break

        if ep % val_every == 0 or ep == max_epochs:
            tau = validate(model, val_data)
            elapsed = time.perf_counter() - t_start
            curve.append((ep, elapsed, tau))
            if first_cross is None and tau >= THRESHOLD:
                first_cross = (ep, elapsed)
            if ep >= 60 and tau < 0.05:
                status = 'diverged'      # stuck at random ranking: stop early
                break

    if not curve:
        return {'lr': lr, 'batch': batch, 'status': status}, curve
    taus = [c[2] for c in curve]
    row = {
        'lr': lr, 'batch': batch, 'status': status,
        'best_tau': max(taus), 'best_ep': curve[int(np.argmax(taus))][0],
        'final_tau': taus[-1], 'final_ep': curve[-1][0],
        't80_ep': first_cross[0] if first_cross else np.nan,
        't80_sec': round(first_cross[1], 2) if first_cross else np.nan,
        'train_sec': round(curve[-1][1], 2),
    }
    return row, curve


def run_screening(args):
    emb_train, label, val_data = load_shared()
    lrs = [float(x) for x in args.only_lrs.split(',') if x.strip()] \
        if args.only_lrs else LRS
    batches = [int(x) for x in args.only_batches.split(',') if x.strip()] \
        if args.only_batches else BATCHES

    done = {}
    if os.path.exists(SWEEP_CSV):
        done = {(r.lr, r.batch): r for r in pd.read_csv(SWEEP_CSV).itertuples()}

    print(f'lr_batch_sweep.py  device={DEVICE}  seed={SEED}  '
          f'max_epochs={args.max_epochs}  val_every={args.val_every}  '
          f'threshold={args.threshold}  grid={len(lrs)}x{len(batches)}')
    rows, curves = [], {}
    for lr in lrs:
        for batch in batches:
            if (lr, batch) in done:
                print(f'  [skip] lr={lr} batch={batch} (already done)')
                rows.append(done[(lr, batch)])
                continue
            t0 = time.time()
            row, curve = train_combo(lr, batch, emb_train, label, val_data,
                                     args.max_epochs, args.val_every)
            rows.append(row)
            curves[(lr, batch)] = curve
            print(f'  lr={lr:<6} batch={batch:>3}  {row["status"]:<8}  '
                  f'best={row["best_tau"]:.4f}  final={row["final_tau"]:.4f}  '
                  f't80={row["t80_sec"]}s  train={row["train_sec"]:6.1f}s  '
                  f'[{time.time() - t0:5.1f}s]')
    df = pd.DataFrame(rows)
    df.to_csv(SWEEP_CSV, index=False)

    # per-combination curves for later figures
    curve_rows = [{'lr': lr, 'batch': batch, 'epoch': c[0], 'sec': c[1],
                   'tau': c[2]} for (lr, batch), cs in curves.items()
                  for c in cs]
    if curve_rows:
        pd.DataFrame(curve_rows).to_csv(CURVES_CSV, index=False)

    analyze(df, args.threshold)
    plot_heatmap(df)
    plot_pareto(df)
    print(f'\nsaved: {SWEEP_CSV}')
    print('figures: lr_batch_heatmap.png, lr_batch_pareto.png')


def analyze(df, threshold):
    """Quality gate + Pareto frontier on (final tau up, time down)."""
    ok = df[df['status'] == 'ok'].copy()
    print(f'\n=== screening summary ({len(ok)}/{len(df)} converged) ===')
    gate = ok[ok['final_tau'] >= threshold].sort_values('final_tau', ascending=False)
    print(f'  gate-passers (final tau >= {threshold}): {len(gate)}')
    if not len(gate):
        print('  no combination passed the gate; widening to best tau >= threshold')
        gate = ok[ok['best_tau'] >= threshold].sort_values('best_tau', ascending=False)
    if not len(gate):
        gate = ok.sort_values('best_tau', ascending=False).head(6)
        print('  no combination reaches the target; showing the top 6')

    # Pareto front: dominated if another combo has tau >= and time <= (both strict once)
    front = []
    g = gate.reset_index(drop=True)
    for i, r in g.iterrows():
        dominated = any((o['final_tau'] >= r['final_tau'] and
                         o['t80_sec'] <= r['t80_sec'] and
                         o['train_sec'] <= r['train_sec'] and
                         (o['final_tau'] > r['final_tau'] or
                          o['t80_sec'] < r['t80_sec'] or
                          o['train_sec'] < r['train_sec']))
                        for _, o in g.iterrows())
        if not dominated:
            front.append(r)
    front_df = pd.DataFrame(front).sort_values('train_sec')
    print('\n  Pareto front (final tau vs time):')
    print(front_df[['lr', 'batch', 'final_tau', 'best_tau', 't80_sec',
                    'train_sec']].to_string(index=False))
    front_df.to_csv(os.path.join(OUT_DIR, 'lr_batch_pareto_front.csv'),
                    index=False)


def plot_heatmap(df):
    piv = df.pivot(index='lr', columns='batch', values='final_tau')
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    im = ax.imshow(piv.values, cmap='viridis', aspect='auto',
                   vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)),
                  [f'{x:g}' for x in piv.index])
    ax.set_xlabel('Batch size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning rate', fontsize=12, fontweight='bold')
    ax.set_title('LP-RCNN: final validation Kendall $\\tau$ on '
                 'BA(2000,4) (lr $\\times$ batch)',
                 fontsize=12.5, fontweight='bold')
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f'{v:.3f}' if np.isfinite(v) else 'div',
                    ha='center', va='center', fontsize=9,
                    color='white' if (np.isfinite(v) and v < 0.55) else 'black')
    # mark the selected setting (lr=0.005, batch=32)
    if 0.005 in piv.index and 32 in piv.columns:
        i = list(piv.index).index(0.005)
        j = list(piv.columns).index(32)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor='crimson', lw=2.5))
    fig.colorbar(im, ax=ax, label='final validation $\\tau$')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'lr_batch_heatmap.png'), dpi=160)
    plt.close(fig)


def plot_pareto(df, threshold=THRESHOLD):
    ok = df[df['status'] == 'ok']
    fig, ax = plt.subplots(figsize=(8, 5.6))
    for lr, grp in ok.groupby('lr'):
        ax.scatter(grp['train_sec'], grp['final_tau'], s=70, label=f'lr={lr:g}')
    for _, r in ok.iterrows():
        ax.annotate(f"{r['lr']:g}/{int(r['batch'])}",
                    (r['train_sec'], r['final_tau']),
                    textcoords='offset points', xytext=(4, 4), fontsize=8)
    ax.axhline(threshold, color='k', ls='--', lw=1.1)
    ax.text(0.98, threshold + 0.008, f'target $\\tau$ = {threshold}',
            transform=ax.get_yaxis_transform(), ha='right', fontsize=9)
    ax.set_xlabel('Wall-clock training time (s, screening budget)', fontsize=12,
                  fontweight='bold')
    ax.set_ylabel('Final validation Kendall $\\tau$ (BA(2000,4))',
                  fontsize=12, fontweight='bold')
    ax.set_title('LP-RCNN lr $\\times$ batch sweep: efficiency vs result',
                 fontsize=12.5, fontweight='bold')
    ax.grid(alpha=0.3, ls='--')
    ax.legend(fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'lr_batch_pareto.png'), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 2 -- confirmation on the nine real-world networks (paper protocol)
# ---------------------------------------------------------------------------

def run_confirmation(combos, seeds, epochs=500):
    emb_train, label, val_data = load_shared()
    print(f'\n=== confirmation on the held-out synthetic network '
          f'{VAL_NET_NAME} ({epochs} epochs, seeds {seeds}) ===')
    rows = []
    for combo in combos:
        lr, batch = combo
        taus, times = [], []
        for seed in seeds:
            Utils.setup_seed(seed)
            loader = build_loader(emb_train, label, batch)
            model = Models.DC_RCNN_MS_DIL(L, in_channels=1).to(DEVICE)
            opt = optim.Adam(model.parameters(), lr=lr)
            t0 = time.time()
            for _ in range(epochs):
                model.train()
                for data, tgt in loader:
                    data = data.to(DEVICE)
                    tgt = tgt.float().to(DEVICE)
                    opt.zero_grad()
                    loss = Utils.listmle_loss(model(data), tgt)
                    loss.backward()
                    opt.step()
            train_sec = time.time() - t0
            # validation score only -- the real-world benchmark stays untouched
            val_tau = validate(model, val_data)
            taus.append(val_tau)
            times.append(train_sec)
            print(f'  lr={lr:g} batch={batch:>3} seed={seed:>2}  '
                  f'val-tau={val_tau:.4f}  train={train_sec:6.1f}s')
        rows.append({'lr': lr, 'batch': batch,
                     'val_tau_mean': float(np.mean(taus)),
                     'val_tau_std': float(np.std(taus)),
                     'train_sec_mean': float(np.mean(times)),
                     'n_seeds': len(seeds)})
    df = pd.DataFrame(rows)
    df.to_csv(CONFIRM_CSV, index=False)
    print('\n=== confirmation summary (validation network '
          f'{VAL_NET_NAME}) ===')
    print(df.sort_values('val_tau_mean', ascending=False).to_string(index=False))
    print(f'saved: {CONFIRM_CSV}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='lr x batch sweep for LP-RCNN')
    ap.add_argument('--max-epochs', type=int, default=EPOCHS)
    ap.add_argument('--val-every', type=int, default=VAL_EVERY)
    ap.add_argument('--threshold', type=float, default=THRESHOLD)
    ap.add_argument('--only-lrs', type=str, default='')
    ap.add_argument('--only-batches', type=str, default='')
    ap.add_argument('--confirm', type=str, default='',
                    help='comma-separated lr/batch pairs, e.g. 0.005/8,0.002/16')
    ap.add_argument('--epochs-confirm', type=int, default=500)
    ap.add_argument('--seeds', type=str, default=str(SEED))
    ap.add_argument('--tag', type=str, default='',
                    help='suffix for the output CSVs (grid sharding)')
    args = ap.parse_args()

    if args.tag:
        global SWEEP_CSV, CONFIRM_CSV, CURVES_CSV
        SWEEP_CSV = os.path.join(OUT_DIR, f'lr_batch_sweep{args.tag}.csv')
        CONFIRM_CSV = os.path.join(OUT_DIR, f'lr_batch_confirm{args.tag}.csv')
        CURVES_CSV = os.path.join(OUT_DIR, f'lr_batch_curves{args.tag}.csv')

    if args.confirm:
        combos = []
        for tok in args.confirm.split(','):
            tok = tok.strip()
            if not tok:
                continue
            lr_s, b_s = tok.split('/')
            combos.append((float(lr_s), int(b_s)))
        seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
        run_confirmation(combos, seeds, epochs=args.epochs_confirm)
        return

    run_screening(args)


if __name__ == '__main__':
    main()
