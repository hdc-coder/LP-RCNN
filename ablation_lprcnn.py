# -*- coding: utf-8 -*-
"""Experiment 2 -- ablation of the single-channel LP-RCNN.

One change per variant, everything else fixed (BA(1000,4), L = 28, lr 0.005,
batch 32, 500 epochs, seed 10):

    LP-RCNN                ND, dilated multi-scale, listMLE
    LP-RCNN (degree)       plain degree channel
    LP-RCNN (no multi)     single 5x5 kernel
    LP-RCNN (flatten head) flattened head instead of GAP
    LP-RCNN (MSE)          squared loss instead of listMLE
"""
import os
import argparse
import warnings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from scipy import stats
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

import Utils
import Embeddings
import Models
import evaluate
from evaluate import NetworkSet, BETA_RATIOS, BETA_1_6_IDX

DEVICE = evaluate.DEVICE
L = evaluate.L
LR = 0.005
BATCH = 32
EPOCHS = 500
SEED = 10

MODEL_DIR = os.path.join('ablation_models')
OUT_DIR = os.path.join('figures', 'ablation')
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_NET_PATH = './Networks/training/Train_1000_4.txt'
TRAIN_SIR_PREFIX = './SIR results/Train_1000_4/BA_1000_4_'


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

def build_variant_specs():
    """{variant_name: (model builder, embedding fn, n_channels, use_listmle, cache_key)}."""
    return {
        'LP-RCNN': (
            lambda: Models.DC_RCNN_MS_DIL(L, in_channels=1),
            lambda G: Embeddings.main_ndrcnn(G, L), 1, True, 'ms1ch_nd'),
        'LP-RCNN (degree)': (
            lambda: Models.DC_RCNN_MS_DIL(L, in_channels=1),
            lambda G: Embeddings.main(G, L), 1, True, 'ms1ch_degree'),
        'LP-RCNN (no multi)': (
            lambda: Models.DC_RCNN(L, in_channels=1),
            lambda G: Embeddings.main_ndrcnn(G, L), 1, True, 'no_multi'),
        'LP-RCNN (flatten head)': (
            lambda: Models.DC_RCNN_MS_DIL_Flat(L, in_channels=1),
            lambda G: Embeddings.main_ndrcnn(G, L), 1, True, 'flatten_head'),
        'LP-RCNN (MSE)': (
            lambda: Models.DC_RCNN_MS_DIL(L, in_channels=1),
            lambda G: Embeddings.main_ndrcnn(G, L), 1, False, 'ms1ch_mse'),
    }


# one ordered style list, so a variant can never drop out of a figure
VARIANT_STYLE = [
    ('LP-RCNN',                'crimson',    's', 3.0),
    ('LP-RCNN (no multi)',     'royalblue',  '^', 2.4),
    ('LP-RCNN (degree)',       'darkgreen',  'o', 2.4),
    ('LP-RCNN (MSE)',          'darkorange', 'v', 2.4),
    ('LP-RCNN (flatten head)', 'purple',     'D', 2.4),
]
assert {v for v, _, _, _ in VARIANT_STYLE} == set(build_variant_specs()), \
    'VARIANT_STYLE must cover every variant returned by build_variant_specs()'


# ---------------------------------------------------------------------------
# Training (cached to ablation_models/)
# ---------------------------------------------------------------------------

def _build_loader(emb, label, C):
    nodes = list(emb.keys())
    X = torch.zeros(len(nodes), C, L, L)
    y = torch.zeros(len(nodes), 1)
    for i, node in enumerate(nodes):
        m = emb[node]
        X[i] = m if isinstance(m, torch.Tensor) else torch.from_numpy(np.asarray(m, dtype=np.float32))
        y[i, 0] = label.get(node, 0.0)
    return DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=True)


def _train(model, loader, use_listmle):
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(EPOCHS):
        for data, tgt in loader:
            data = data.to(DEVICE)
            tgt = tgt.float().to(DEVICE)
            opt.zero_grad()
            pred = model(data)
            loss = Utils.listmle_loss(pred, tgt) if use_listmle else nn.MSELoss()(pred, tgt)
            loss.backward()
            opt.step()
    return model


def get_variant(name, model_builder, emb_fn, C, use_listmle, cache_key, force=False):
    """Load a cached variant or train (and cache) it from scratch."""
    path = os.path.join(MODEL_DIR, f'{cache_key}_ep{EPOCHS}.pt')
    if os.path.exists(path) and not force:
        print(f'  [load] {name} <- {path}')
        model = model_builder()
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        return model

    print(f'  [train] {name}  ({EPOCHS} epochs, {"listMLE" if use_listmle else "MSE"})')
    G = Utils.load_graph(TRAIN_NET_PATH)
    label = evaluate.train_label(TRAIN_SIR_PREFIX)
    emb = emb_fn(G)
    loader = _build_loader(emb, label, C)
    model = _train(model_builder(), loader, use_listmle)
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_variant(model, emb, nodes, C, sir_list):
    scores = evaluate.model_scores(model, emb, nodes, C)
    return evaluate.kendall_taus(scores, nodes, sir_list)


# ---------------------------------------------------------------------------
# Figures and table
# ---------------------------------------------------------------------------

def plot_curves(tau_dict, networks):
    n_nets = len(networks)
    ncols = 4 if n_nets > 10 else 3
    nrows = int(np.ceil(n_nets / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, net in enumerate(networks):
        ax = axes[i]
        for variant, c, m, lw in VARIANT_STYLE:
            ax.plot(BETA_RATIOS, tau_dict[variant][net], marker=m, c=c,
                    label=variant, lw=lw,
                    zorder=5 if variant == 'LP-RCNN' else 2)
        ax.set_title(net, fontsize=11, fontweight='bold')
        ax.set_xticks([1.0, 1.3, 1.6, 2.0])
        ax.set_yticks(np.round(np.arange(0.0, 1.01, 0.2), 1))
        ax.set_xlim(1.0, 2.0)
        ax.grid(alpha=0.3, ls='--')
        if i % ncols == 0:
            ax.set_ylabel(r'Kendall $\tau$', fontsize=11)
        if i >= n_nets - ncols:
            ax.set_xlabel(r'$\beta/\beta_{th}$', fontsize=11)
    for j in range(n_nets, len(axes)):
        axes[j].axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(tau_dict),
               fontsize=9, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle('Ablation study: Kendall tau-b vs beta/beta_th',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    path = os.path.join(OUT_DIR, 'ablation_tau_curves.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f'saved: {path}')


def plot_average(tau_dict, networks):
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant, c, m, lw in VARIANT_STYLE:
        by_net = tau_dict[variant]
        avg = np.mean([by_net[net] for net in networks], axis=0)
        ax.plot(BETA_RATIOS, avg, marker=m, c=c, label=variant, lw=lw,
                zorder=5 if variant == 'LP-RCNN' else 2)
    ax.set_xlabel(r'$\beta/\beta_{th}$', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Kendall ' + r'$\tau$', fontsize=13, fontweight='bold')
    ax.set_title('Ablation: average Kendall tau-b over {} networks'.format(len(networks)),
                 fontsize=13, fontweight='bold')
    ax.set_xticks([1.0, 1.3, 1.6, 2.0])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.round(np.arange(0.0, 1.01, 0.2), 1))
    ax.grid(alpha=0.3, ls='--')
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'ablation_tau_average.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f'saved: {path}')


def save_table_at_beta1_6(tau_dict, networks):
    rows = {variant: {net: float(by_net[net][BETA_1_6_IDX]) for net in networks}
            for variant, by_net in tau_dict.items()}
    df = pd.DataFrame(rows).T
    df.index.name = 'Variant'
    df['Mean'] = df.mean(axis=1)
    df = df.sort_values('Mean', ascending=False)

    path = os.path.join(OUT_DIR, 'table_beta1_6.csv')
    df.to_csv(path)
    print('\n=== Ablation: Kendall tau-b at beta/beta_th = 1.6 ===')
    evaluate.print_table(df)
    print(f'\nsaved: {path}')
    return df


def save_tau_curves(tau_dict, networks):
    """Long-format curves, so the per-ratio evidence can be recomputed.
    """
    rows = []
    for variant, by_net in tau_dict.items():
        for net in networks:
            for ratio, t in zip(BETA_RATIOS, by_net[net]):
                rows.append({'Variant': variant, 'Network': net,
                             'beta_ratio': float(ratio), 'tau': float(t)})
    path = os.path.join(OUT_DIR, 'table_tau_curves.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'saved: {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='LP-RCNN ablation study')
    ap.add_argument('--networks', type=str, default='')
    ap.add_argument('--force', action='store_true', help='retrain all variants')
    args = ap.parse_args()

    global OUT_DIR
    Utils.setup_seed(SEED)
    set_ = NetworkSet()
    if args.networks:
        sel = [s.strip() for s in args.networks.split(',') if s.strip()]
        networks = [n for n in set_.names() if n in sel]
        if networks != set_.names():
            OUT_DIR = os.path.join('figures', 'ablation',
                                   'subset_' + '_'.join(networks))
            os.makedirs(OUT_DIR, exist_ok=True)
            print(f'[subset run] outputs redirected to {OUT_DIR} so the full '
                  f'ablation tables are not overwritten')
    else:
        networks = set_.names()

    print(f'ablation_lprcnn.py  device={DEVICE}  networks={networks}')

    specs = build_variant_specs()
    print('Training / loading variants ...')
    models = {}
    for variant, (model_builder, emb_fn, C, use_listmle, cache_key) in specs.items():
        models[variant] = (get_variant(variant, model_builder, emb_fn, C, use_listmle,
                                       cache_key, force=args.force), emb_fn, C)
        torch.cuda.empty_cache()

    print('\nEvaluating variants on the real-world networks ...')
    tau_dict = {variant: {} for variant in specs}
    for i, net in enumerate(networks):
        G, sir_list = set_.graph(net), set_.sir(net)
        print(f'[{i + 1}/{len(networks)}] {net} (N={G.number_of_nodes()})')
        nodes = list(G.nodes())
        for variant, (model, emb_fn, C) in models.items():
            emb = emb_fn(G)
            tau_dict[variant][net] = eval_variant(model, emb, nodes, C, sir_list)
        print('   done')

    mean_df = evaluate.build_table(tau_dict, networks)
    print('\n=== Ablation: average Kendall tau-b over beta/beta_th = 1.0..2.0 ===')
    evaluate.print_table(mean_df)
    mean_df.to_csv(os.path.join(OUT_DIR, 'table_mean_tau.csv'))

    # paired Wilcoxon signed-rank test of each deletion across the networks
    network_cols = [c for c in mean_df.columns if c not in ('Mean', '#Best')]
    base = mean_df.loc['LP-RCNN', network_cols]
    p_rows = []
    for variant in mean_df.index:
        if variant == 'LP-RCNN':
            continue
        diff = base - mean_df.loc[variant, network_cols]
        stat, p = stats.wilcoxon(diff)
        p_rows.append({'variant': variant, 'wilcoxon_statistic': stat,
                       'p_value': p, 'mean_delta': float(diff.mean())})
    p_df = pd.DataFrame(p_rows)
    p_df.to_csv(os.path.join(OUT_DIR, 'ablation_pvalues.csv'), index=False)
    print('\n=== Ablation significance (Wilcoxon signed-rank over networks) ===')
    print(p_df.round(6).to_string(index=False))


    plot_curves(tau_dict, networks)
    plot_average(tau_dict, networks)
    save_table_at_beta1_6(tau_dict, networks)
    save_tau_curves(tau_dict, networks)


if __name__ == '__main__':
    main()
