# -*- coding: utf-8 -*-
"""Experiment 3b -- epochs needed to reach a target Kendall tau_b.

Three single-variable contrasts on BA(1000,4) with the ND channel, lr 0.005,
batch 32 and no early stopping:

    head       flatten head vs GAP head   (both ND, MSE)
    loss       MSE vs listMLE             (both GAP, ND)
    front end  5x5 vs dilated 3/5/7       (both GAP, listMLE)

Three seeds each, validated every ``--val-every`` epochs on Email, Ham, NS and
Facebook.  The figure plots tau against the epoch with a star at the crossing of
``--threshold``; RCNN / M-RCNN / N-RCNN are reference curves in a second panel.

    python time_to_accuracy.py --max-epochs 300 --seeds 10,11,12
"""
import os
import time
import argparse
import warnings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

import Utils
import Embeddings
import Models
import evaluate
from evaluate import DEVICE, L

SEEDS = [10, 11, 12]
LR = 0.005
BATCH = 32
EPOCHS = 200
VAL_EVERY = 5
# target tau_b read off both panels (0.85 splits the contrast curves)
THRESHOLD = 0.85

TRAIN_NET_PATH = './Networks/training/Train_1000_4.txt'
VAL_NETS = ['Email', 'Ham', 'NS', 'Facebook']   # held out from training

OUT_DIR = os.path.join('figures', 'time_to_accuracy')
os.makedirs(OUT_DIR, exist_ok=True)

MSE = nn.MSELoss()


# ---------------------------------------------------------------------------
# Variant matrix: name -> (model builder, embedding kind, C, loss, lr, batch,
#                         group, head kind, input channel)
# ---------------------------------------------------------------------------

def _mrcnn_emb(G):
    _, com, _ = Utils.Louvain(G)
    return Embeddings.main1(G, L, com)


def _nrcnn_emb(G, net_name):
    dc = dict(nx.degree_centrality(G))
    bc = evaluate.betweenness(net_name, G)
    hc = Utils.calcHIndexValues(G)
    cc = evaluate.closeness(net_name, G)
    return Embeddings.main_n_rcnn(G, L, centralities=(dc, bc, hc, cc))


def build_variants():
    """``name: (builder, emb_kind, C, use_listmle, lr, batch, group, head, in_ch)``.
    """
    return {
        # --- published baselines (reference curves, paper protocols) ------
        'RCNN':   (lambda: Models.CNN(L),              'degree', 1, False, 0.001, 32,
                   'reference', 'flatten', 'degree'),
        'M-RCNN': (lambda: Models.CNN1(L),             'mrcnn',  3, False, 0.001, 32,
                   'reference', 'flatten', 'ND/NCom/k-core'),
        'N-RCNN': (lambda: Models.N_RCNN(L),           'nrcnn',  3, False, 0.001, 32,
                   'reference', 'flatten', 'DC/BC/HC'),
        # --- single-variable contrasts (uniform protocol lr=0.005, b=32) ---
        'Flatten head (ND, MSE)':   (lambda: Models.CNN(L),           'nd', 1, False, 0.005, 32,
                                     'contrast', 'flatten', 'ND'),
        'GAP head (ND, MSE)':       (lambda: Models.DC_RCNN(L),       'nd', 1, False, 0.005, 32,
                                     'contrast', 'GAP', 'ND'),
        'GAP head (ND, listMLE)':   (lambda: Models.DC_RCNN(L),       'nd', 1, True,  0.005, 32,
                                     'contrast', 'GAP', 'ND'),
        'LP-RCNN':                   (lambda: Models.DC_RCNN_MS_DIL(L), 'nd', 1, True,  0.005, 32,
                                     'contrast', 'GAP', 'ND'),
    }


def make_emb(kind, G, net_name=''):
    """Build the embedding dict of ``G`` for the given embedding kind."""
    if kind == 'degree':
        return Embeddings.main(G, L)
    if kind == 'nd':
        return Embeddings.main_ndrcnn(G, L)
    if kind == 'mrcnn':
        return _mrcnn_emb(G)
    if kind == 'nrcnn':
        return _nrcnn_emb(G, net_name)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def stack(emb, C):
    """Stack an embedding dict into a [N, C, L, L] tensor + aligned node list."""
    nodes = list(emb.keys())
    X = torch.zeros(len(nodes), C, L, L)
    for i, node in enumerate(nodes):
        m = emb[node]
        X[i] = m if isinstance(m, torch.Tensor) else torch.from_numpy(
            np.asarray(m, dtype=np.float32))
    return X, nodes


def build_train_loader(emb, label, C, batch):
    from torch.utils.data import DataLoader, TensorDataset
    X, nodes = stack(emb, C)
    y = torch.zeros(len(nodes), 1)
    for i, node in enumerate(nodes):
        y[i, 0] = label.get(node, 0.0)
    return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=True)


# ---------------------------------------------------------------------------
# Training with periodic held-out validation (full budget, no early stop)
# ---------------------------------------------------------------------------

def validate(model, val_data):
    """Mean Kendall tau-b of ``model`` over the held-out validation networks."""
    taus = []
    model.eval()
    with torch.no_grad():
        for Xv, nodes_v, sirs in val_data:
            scores = model(Xv.to(DEVICE)).cpu().numpy().flatten()
            taus.append(np.mean(evaluate.kendall_taus(scores, nodes_v, sirs)))
    return float(np.mean(taus))


def train_one_seed(name, builder, C, use_listmle, emb_train, label, val_data,
                   max_epochs, val_every, lr, batch, seed, threshold):
    """Train one seed for the full budget; return the (epoch, tau, sec) curve
    and the first crossing epoch (None if the target is never reached)."""
    Utils.setup_seed(seed)
    loader = build_train_loader(emb_train, label, C, batch)
    model = builder().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = Utils.listmle_loss if use_listmle else MSE
    curve = []
    first_cross = None
    t_start = time.perf_counter()

    for ep in range(1, max_epochs + 1):
        model.train()
        for data, tgt in loader:
            data = data.to(DEVICE)
            tgt = tgt.float().to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(data), tgt)
            loss.backward()
            opt.step()

        if ep % val_every == 0 or ep == max_epochs:
            tau = validate(model, val_data)
            elapsed = time.perf_counter() - t_start
            curve.append((ep, elapsed, tau))
            if first_cross is None and tau >= threshold:
                first_cross = (ep, elapsed)
    return curve, first_cross


# ---------------------------------------------------------------------------
# Aggregation and summary
# ---------------------------------------------------------------------------

def summarize(name, spec, seed_runs, threshold):
    """Mean +- std over seeds; ``seed_runs`` is ``[(curve, first_cross)]``.
    """
    epochs = [c[0] for c in seed_runs[0][0]]
    tau_by_ep = {}
    for ep in epochs:
        vals = []
        for curve, _ in seed_runs:
            for ce, _, ct in curve:
                if ce == ep:
                    vals.append(ct)
        tau_by_ep[ep] = vals
    mean_tau = np.array([float(np.mean(tau_by_ep[ep])) for ep in epochs])
    std_tau = np.array([float(np.std(tau_by_ep[ep])) for ep in epochs])

    crosses = [cc for _, cc in seed_runs if cc is not None]
    cross_ep = np.array([cc[0] for cc in crosses], dtype=float)
    cross_sec = np.array([cc[1] for cc in crosses])
    best = [max(c[2] for c in curve) for curve, _ in seed_runs]
    final = [curve[-1][2] for curve, _ in seed_runs]

    row = {
        'metric': 'tau_b',
        'variant': name, 'group': spec[6], 'head': spec[7],
        'input': spec[8], 'loss': 'listMLE' if spec[3] else 'MSE',
        'lr': spec[4], 'batch': spec[5],
        'n_seeds': len(seed_runs), 'n_reached': len(crosses),
        'epochs_to_target_mean': float(cross_ep.mean()) if len(crosses) else np.nan,
        'epochs_to_target_std': float(cross_ep.std(ddof=1)) if len(crosses) >= 2 else 0.0,
        'seconds_to_target_mean': float(cross_sec.mean()) if len(crosses) else np.nan,
        'seconds_to_target_std': float(cross_sec.std(ddof=1)) if len(crosses) >= 2 else 0.0,
        'best_tau_mean': float(np.mean(best)), 'best_tau_std': float(np.std(best)),
        'final_tau_mean': float(np.mean(final)), 'final_tau_std': float(np.std(final)),
        'mean_curve': mean_tau, 'std_curve': std_tau, 'epochs': epochs,
        'cross_ep': cross_ep, 'cross_sec': cross_sec,
    }
    return row


def _save(fig, path):
    """Write one CSV and echo its path.
    """
    fig.savefig(path, dpi=600)
    vec = os.path.splitext(path)[0] + '.pdf'
    fig.savefig(vec)
    print(f'saved: {path} (+ {vec})')


def plot_curves(rows, threshold, save_path):
    """Two panels: (a) single-variable contrasts, (b) published baselines.
    Mean curve + std band + per-seed thin lines; star at first crossing."""
    STYLE = {
        'Flatten head (ND, MSE)':   dict(c='#4E79A7', m='o', ls='-'),
        'GAP head (ND, MSE)':       dict(c='#B07C3C', m='s', ls='-'),
        'GAP head (ND, listMLE)':   dict(c='#59A14F', m='v', ls='-'),
        'LP-RCNN':                   dict(c='#D1495B', m='s', ls='-', lw=2.4),
        'RCNN':                      dict(c='#4E79A7', m='o', ls='-'),
        'M-RCNN':                    dict(c='#E0B341', m='^', ls='-'),
        'N-RCNN':                    dict(c='#8E6DA8', m='D', ls='--'),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
    panels = [('contrast', 'Single-variable contrasts (identical protocol: '
                            'ND input, lr=0.005, batch=32)'),
              ('reference', 'Published baselines (their paper protocols)')]
    # both panels share one y-range closed at tau_b = 1, so the floor has to
    # reach the N-RCNN plateau while keeping the top curves apart
    CEIL = 1.00
    legend_loc = {'contrast': 'lower left', 'reference': 'lower right'}

    def shared_floor():
        """Lowest round y-limit (0.05 grid) that still contains every curve
        of either panel, never above the target that doubles as a bound."""
        lo = threshold
        for r in rows:
            mu = np.asarray(r['mean_curve'])
            sd = np.asarray(r['std_curve'])
            lo = min(lo, float((mu - sd).min()), float(mu.min()))
            for cs in r['_curves']:
                lo = min(lo, min(c[2] for c in cs))
        lo = float(np.floor((lo - 0.02) / 0.05) * 0.05)
        # fine grid while the span is small enough to keep every label apart
        step = 0.05 if CEIL - lo <= 0.70 else 0.10
        # drop the floor onto the tick grid, so the ceiling is itself a tick
        # instead of leaving an unlabelled strip at the top
        return CEIL - float(np.ceil((CEIL - lo) / step - 1e-9)) * step, step

    FLOOR, YSTEP = shared_floor()
    xmax = max(max(r['epochs']) for r in rows)
    for i, (ax, (group, title)) in enumerate(zip(axes, panels)):
        labels = []
        for r in rows:
            if r['group'] != group:
                continue
            st = STYLE[r['variant']]
            ep = np.array(r['epochs'])
            mu, sd = r['mean_curve'], r['std_curve']
            ax.fill_between(ep, mu - sd, mu + sd, color=st['c'], alpha=0.15)
            # per-seed thin lines (honest spread) -- stored separately below
            for cs in r['_curves']:
                ax.plot([c[0] for c in cs], [c[2] for c in cs],
                        color=st['c'], alpha=0.25, lw=0.8)
            ax.plot(ep, mu, marker=st['m'], c=st['c'], lw=st.get('lw', 1.8),
                    ls=st['ls'], label=r['variant'], zorder=5,
                    markevery=max(1, len(ep) // 12), ms=5)
            if len(r['cross_ep']) > 0:
                cep = r['cross_ep'].mean()
                ax.plot(cep, threshold, marker='*', ms=14, c=st['c'],
                        mec='k', mew=0.6, zorder=6)
            labels.append((float(mu[-1]), st['c']))

        ax.set_ylim(FLOOR, CEIL)
        ax.set_yticks(np.arange(FLOOR, CEIL + 1e-9, YSTEP))

        # push the end-of-curve labels apart, they end within thousandths
        labels.sort(key=lambda t: -t[0])
        span = CEIL - FLOOR
        gap = span * 0.05
        placed = []
        for k, (y, _) in enumerate(labels):
            if k and y > placed[-1] - gap:
                y = placed[-1] - gap
            placed.append(y)
        # three decimals: LP-RCNN and the listMLE variant differ by ~0.003
        for (val, col), y in zip(labels, placed):
            ax.text(xmax, y, f'  {val:.3f}', fontsize=8, va='center',
                    color=col, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2.5,
                                                foreground='w')])

        ax.axhline(threshold, color='k', ls='--', lw=1.2)
        ax.set_title(f'({chr(97 + i)}) {title}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Training epochs on BA(1000,4)', fontsize=11,
                      fontweight='bold')
        ax.set_xlim(0, xmax)
        ax.grid(alpha=0.3, ls='--')
        # the target sits in the legend, not beside the line, where the
        # curves would cover it
        handles, names = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], color='k', ls='--', lw=1.2))
        names.append(f'target $\\tau_b$ = {threshold}')
        ax.legend(handles, names, fontsize=9, loc=legend_loc[group],
                  framealpha=0.9)

    axes[0].set_ylabel('Mean Kendall $\\tau_b$ on held-out networks\n'
                       '(Email, Ham, NS, Facebook)', fontsize=11,
                       fontweight='bold')
    fig.suptitle('Time-to-convergence on BA(1000,4)  (mean $\\pm$ std over '
                 f'{rows[0]["n_seeds"]} seeds; star = first crossing of the target)',
                 fontsize=12.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Time-to-convergence of GAP vs flatten')
    ap.add_argument('--only', type=str, default='',
                    help='comma-separated subset of variants')
    ap.add_argument('--max-epochs', type=int, default=EPOCHS)
    ap.add_argument('--val-every', type=int, default=VAL_EVERY)
    ap.add_argument('--threshold', type=float, default=THRESHOLD)
    ap.add_argument('--seeds', type=str, default=','.join(map(str, SEEDS)))
    ap.add_argument('--val-nets', type=str, default=','.join(VAL_NETS),
                    help='comma-separated held-out real-world networks')
    ap.add_argument('--plot-only', action='store_true',
                    help='regenerate the figure from the saved CSVs (no training)')
    args = ap.parse_args()

    if args.plot_only:
        df = pd.read_csv(os.path.join(OUT_DIR, 'time_to_accuracy.csv'))
        curves = pd.read_csv(os.path.join(OUT_DIR, 'time_to_accuracy_curves.csv'))
        rows = []
        for _, r in df.iterrows():
            sub = curves[curves['variant'] == r['variant']]
            epochs = np.array(sorted(sub['epoch'].unique()))
            mu = np.array([sub[sub['epoch'] == ep]['tau'].mean() for ep in epochs])
            sd = np.array([sub[sub['epoch'] == ep]['tau'].std() for ep in epochs])
            per_seed = []
            for seed in sorted(sub['seed'].unique()):
                s = sub[sub['seed'] == seed].sort_values('epoch')
                per_seed.append([(row.epoch, row.seconds, row.tau)
                                 for row in s.itertuples()])
            # Recompute the crossing from the curves instead of reading the
            # stored mean, so a star always sits on the target line that is
            # actually drawn (the two disagree once --threshold is changed).
            cross = np.array(
                [min(c[0] for c in cs if c[2] >= args.threshold)
                 for cs in per_seed
                 if any(c[2] >= args.threshold for c in cs)], dtype=float)
            rows.append({'variant': r['variant'], 'group': r['group'],
                         'epochs': epochs, 'mean_curve': mu, 'std_curve': sd,
                         '_curves': per_seed, 'cross_ep': cross,
                         'n_seeds': int(r['n_seeds'])})
        plot_curves(rows, args.threshold,
                    os.path.join(OUT_DIR, 'time_to_accuracy.png'))
        print('saved: figures/time_to_accuracy/time_to_accuracy.png')
        return

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    val_nets = [n.strip() for n in args.val_nets.split(',') if n.strip()]
    print(f'time_to_accuracy.py (v2)  device={DEVICE}  seeds={seeds}  '
          f'max_epochs={args.max_epochs}  val_every={args.val_every}  '
          f'target_tau={args.threshold}  val_nets={val_nets}')

    variants = build_variants()
    if args.only:
        # Variant names such as ``Flatten head (ND, MSE)`` contain a comma,
        # so accept either one exact variant name or a comma-separated list.
        if args.only in variants:
            sel = [args.only]
        else:
            sel = [s.strip() for s in args.only.split(',') if s.strip()]
        variants = {k: v for k, v in variants.items() if k in sel}

    print('\nLoading training data (BA(1000,4)) ...')
    import train as train_mod
    G, label = train_mod.load_training_data()

    print('Loading held-out validation networks ...')
    set_ = evaluate.NetworkSet()
    val_sets = {n: set_.graph(n) for n in val_nets}

    rows, curves_by_variant = [], {}
    for name, spec in variants.items():
        builder, emb_kind, C, use_listmle, lr, batch = spec[:6]
        print(f'\n[{name}] building embeddings (train BA1000 + '
              f'{len(val_nets)} val nets) ...')
        emb_train = make_emb(emb_kind, G, 'BA(1000,4)')

        # precompute every validation embedding once (per variant) -- the
        # per-checkpoint cost is then only forward + kendall-tau.
        val_data = []
        for net in val_nets:
            Gv, sirs = val_sets[net], set_.sir(net)
            Xv, nodes_v = stack(make_emb(emb_kind, Gv, net), C)
            missing = [n for n in nodes_v if n not in sirs[0]]
            assert not missing, f'{net}: {len(missing)} nodes lack SIR labels'
            val_data.append((Xv, nodes_v, sirs))
        print(f'  validation sets ready: '
              f'{[(net, val_sets[net].number_of_nodes()) for net in val_nets]}')

        print(f'  training {len(seeds)} seeds '
              f'(loss={"listMLE" if use_listmle else "MSE"}, lr={lr}, '
              f'batch={batch}) ...')
        seed_runs = []
        for seed in seeds:
            t0 = time.time()
            curve, first_cross = train_one_seed(
                name, builder, C, use_listmle, emb_train, label, val_data,
                args.max_epochs, args.val_every, lr, batch, seed,
                args.threshold)
            seed_runs.append((curve, first_cross))
            best = max(c[2] for c in curve)
            cross = f'crossed@ep{first_cross[0]}' if first_cross else 'no-cross'
            print(f'    seed {seed:>2}  best tau={best:.4f}  final={curve[-1][2]:.4f}'
                  f'  {cross}  [{time.time() - t0:5.1f}s]')

        row = summarize(name, spec, seed_runs, args.threshold)
        row['_curves'] = [c for c, _ in seed_runs]
        curves_by_variant[name] = [c for c, _ in seed_runs]
        rows.append(row)

    # summary table (mean +- std over seeds)
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in
                        ('mean_curve', 'std_curve', 'epochs',
                         'cross_ep', 'cross_sec', '_curves')} for r in rows])
    csv_path = os.path.join(OUT_DIR, 'time_to_accuracy.csv')
    df.to_csv(csv_path, index=False)

    # per-seed curves for reproducibility
    curve_rows = [{'variant': name, 'seed': seed, 'epoch': c[0],
                   'seconds': round(c[1], 2), 'tau': c[2]}
                  for name, seed_curves in curves_by_variant.items()
                  for seed, cs in zip(seeds, seed_curves) for c in cs]
    pd.DataFrame(curve_rows).to_csv(
        os.path.join(OUT_DIR, 'time_to_accuracy_curves.csv'), index=False)

    print(f'\n=== time to reach target tau = {args.threshold} '
          f'({len(seeds)} seeds, mean +- std) ===')
    for r in rows:
        if r['n_reached']:
            print(f"  {r['variant']:>24}  ep {r['epochs_to_target_mean']:6.1f} "
                  f"+/- {r['epochs_to_target_std']:4.1f}   "
                  f"{r['seconds_to_target_mean']:6.1f} +/- "
                  f"{r['seconds_to_target_std']:4.1f} s   "
                  f"({r['n_reached']}/{r['n_seeds']} seeds)")
        else:
            print(f"  {r['variant']:>24}  NOT reached "
                  f"(best {r['best_tau_mean']:.3f} +/- {r['best_tau_std']:.3f}, "
                  f"final {r['final_tau_mean']:.3f} +/- {r['final_tau_std']:.3f})")

    plot_curves(rows, args.threshold,
                os.path.join(OUT_DIR, 'time_to_accuracy.png'))
    print(f'\nsaved: {csv_path}')
    print('saved: figures/time_to_accuracy/time_to_accuracy.png')


if __name__ == '__main__':
    main()
