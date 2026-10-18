# -*- coding: utf-8 -*-
"""Experiment 1 -- LP-RCNN against the state of the art.

Kendall tau-b against SIR, nine networks, eleven ratios, for LP-RCNN, RCNN
(Yu 2020), M-RCNN (Ou 2022), N-RCNN, AGNN (2024) and the classical indices
BC / DC / Vc / K-core / ND.  STGCN is absent: our reproduction scores ~0.42,
below DC and k-core.

Writes ``figures/compare/``: two figures and four CSVs.
"""
import os
import argparse
import warnings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import networkx as nx
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.optim as optim

import Utils
import Embeddings
import Models
import gnn_models
import evaluate
from evaluate import NetworkSet, BETA_RATIOS, BETA_1_6_IDX

OUT_DIR = os.path.join('figures', 'compare')
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = ['LP-RCNN', 'RCNN', 'M-RCNN', 'N-RCNN', 'AGNN',
           'BC', 'DC', 'Vc', 'K-core', 'ND']

# one marker/colour per method; LP-RCNN is red and slightly thicker
STYLE = {
    'LP-RCNN': dict(marker='s', c='#d62728', ms=6, lw=2.6, ls='-', zorder=5,
                    markevery=1),
    'RCNN':    dict(marker='o', c='#1f77b4', ms=5.5, lw=1.7, ls='-', zorder=3,
                    markevery=1),
    'M-RCNN':  dict(marker='^', c='#ff7f0e', ms=5.5, lw=1.7, ls='-', zorder=3,
                    markevery=1),
    'N-RCNN':  dict(marker='D', c='#2ca02c', ms=5.5, lw=1.7, ls='--', zorder=3,
                    markevery=1),
    'AGNN':    dict(marker='P', c='#9467bd', ms=6, lw=1.7, ls='-', zorder=3,
                    markevery=1),
    'BC':      dict(marker='v', c='#8c564b', ms=5.5, lw=1.7, ls='-', zorder=3,
                    markevery=1),
    'DC':      dict(marker='<', c='#e377c2', ms=5.5, lw=1.7, ls='-', zorder=3,
                    markevery=1),
    'Vc':      dict(marker='>', c='#7f7f7f', ms=5.5, lw=1.7, ls='-', zorder=3,
                    markevery=1),
    'K-core':  dict(marker='*', c='#bcbd22', ms=7, lw=1.7, ls='-', zorder=3,
                    markevery=1),
    'ND':      dict(marker='h', c='#17becf', ms=6, lw=1.7, ls='-', zorder=3,
                    markevery=1),
}


# --- Model loading ---------------------------------------------------------

def _train_on_ba(name, build_model, emb_fn, C, use_listmle):
    """Train one model on BA(1000, 4) when no cached weights are available."""
    import train as train_mod
    Utils.setup_seed(10)   # fixed seed, matching train.py's default
    G, label = train_mod.load_training_data()
    emb = emb_fn(G, evaluate.L)
    # the baselines keep their papers' lr 1e-3; RCNN's paper omits the batch
    # size, so the family value 32 is used
    batch = 32
    lr = 0.005 if name == 'LP-RCNN' else 0.001
    loader_fn = {1: Utils.Get_DataLoader, 3: Utils.Get_DataLoader1}[C]
    loader = loader_fn(emb, label, batch, evaluate.L)
    model = build_model()
    if use_listmle:
        Utils.train_model_listmle(loader, model, 500, lr, evaluate.L)
    else:
        Utils.train_model(loader, model, 500, lr, evaluate.L)
    return model


def _gnn_train(name):
    """Train AGNN/STGCN on BA(1000, 4) under their official pipelines."""
    import train as train_mod
    G, label = train_mod.load_training_data()
    nodes = list(G.nodes())
    dev = evaluate.DEVICE
    y = torch.zeros(len(nodes), 1)
    for i, node in enumerate(nodes):
        y[i, 0] = label.get(node, 0.0)
    y = y.to(dev).squeeze(1)

    if name == 'AGNN':
        Utils.setup_seed(10)   # fixed seed for the BA-side training
        A_norm_sp = gnn_models.sparse_anorm(G, nodes).to(dev)
        A_sp, D_ba = gnn_models.sparse_ai(G, nodes)
        A_sp, D_ba = A_sp.to(dev), D_ba.to(dev)
        ae_ba = gnn_models.AgnAE(len(nodes)).to(dev)
        gnn_models.train_agnn_ae(ae_ba, A_norm_sp, D_ba)
        Z_ba = ae_ba.latent(A_norm_sp).detach()
        model = gnn_models.AgnRank().to(dev)
        gnn_models.train_agnn_rank(model, A_sp, Z_ba, y)
        return model


    A_norm, D = gnn_models.build_graph(G, nodes)
    A_norm, D = A_norm.to(dev), D.to(dev)
    members = []
    for seed in range(10, 25):
        Utils.setup_seed(seed)
        m = gnn_models.STGCN().to(dev)
        gnn_models.train_stgcn(m, A_norm, D, y)
        members.append(m)
    return gnn_models.Ensemble(members)


def ensure_model(name, build_model, emb_fn, C, use_listmle, full_obj_paths, state_dict_paths):
    """Load a trained model from the first existing path, otherwise train it."""
    for p in full_obj_paths:
        if os.path.exists(p):
            print(f'  [load] {name} <- {p}')
            return torch.load(p, map_location=evaluate.DEVICE, weights_only=False)
    for p in state_dict_paths:
        if os.path.exists(p):
            print(f'  [load] {name} <- {p}')
            model = build_model()
            model.load_state_dict(torch.load(p, map_location=evaluate.DEVICE))
            return model
    print(f'  [train] {name} from scratch on BA(1000, 4)')
    return _train_on_ba(name, build_model, emb_fn, C, use_listmle)


def _mrcnn_emb(G, L):
    _, com, _ = Utils.Louvain(G)
    return Embeddings.main1(G, L, com)


def load_models():
    """Return the trained CNN models used in the comparison."""
    print('Loading neural-network models ...')
    models = {
        'LP-RCNN': ensure_model(
            'LP-RCNN', lambda: Models.DC_RCNN_MS_DIL(evaluate.L, in_channels=1),
            Embeddings.main_ndrcnn, 1, True,
            ['Models/lprcnn.pt'], []),
        'RCNN': ensure_model(
            'RCNN', lambda: Models.CNN(evaluate.L), Embeddings.main, 1, False,
            ['Models/rcnn.pt', 'Models/RCNN28-2'], []),
        'M-RCNN': ensure_model(
            'M-RCNN', lambda: Models.CNN1(evaluate.L), _mrcnn_emb, 3, False,
            ['Models/mrcnn.pt', 'Models/MRCNN28-2'], []),
        'N-RCNN': ensure_model(
            'N-RCNN', lambda: Models.N_RCNN(evaluate.L), Embeddings.main_n_rcnn, 3, False,
            ['Models/nrcnn.pt', 'Models/NRCNN28-2'], []),
    }
    # full-graph GNN baselines (AGNN 2024; STGCN stays parked, see module
    # docstring)
    for gname, gpath in [('AGNN', 'Models/agnn.pt')]:
        if gname not in METHODS:
            continue
        if os.path.exists(gpath):
            print(f'  [load] {gname} <- {gpath}')
            models[gname] = torch.load(gpath, map_location=evaluate.DEVICE, weights_only=False)
        else:
            print(f'  [train] {gname} from scratch on BA(1000, 4)')
            models[gname] = _gnn_train(gname)
    return models


# --- Evaluation ------------------------------------------------------------

def eval_neural(model, name, net, G, set_, sir_list, seed_stats=None):
    """Kendall-tau list for a trained CNN model.

    With ``seed_stats`` and ``name == 'AGNN'``, append the per-AE-seed means.
    """
    nodes = list(G.nodes())
    if name == 'M-RCNN':
        emb = Embeddings.main1(G, evaluate.L, set_.community(net))
        C = 3
    elif name == 'LP-RCNN':
        emb, C = Embeddings.main_ndrcnn(G, evaluate.L), 1
    elif name == 'RCNN':
        emb, C = Embeddings.main(G, evaluate.L), 1
    elif name == 'N-RCNN':
        # cached / parallel BC and CC: computing them inside the embedding
        # costs O(nm) per network
        dc = dict(nx.degree_centrality(G))
        bc = evaluate.betweenness(net, G)
        hc = Utils.calcHIndexValues(G)
        cc = evaluate.closeness(net, G)
        emb, C = Embeddings.main_n_rcnn(G, evaluate.L, centralities=(dc, bc, hc, cc)), 3
    elif name in ('AGNN', 'STGCN'):
        # full-graph baseline; AGNN scores each network's own AE latent with
        # the BA-trained rank module, averaged over 5 AE inits (cached)
        if name == 'AGNN':
            A_norm_sp = gnn_models.sparse_anorm(G, nodes).to(evaluate.DEVICE)
            A_sp, _ = gnn_models.sparse_ai(G, nodes)
            A_sp = A_sp.to(evaluate.DEVICE)
            scores, seed_taus = [], []
            for seed in range(5):
                ae = gnn_models.load_or_train_agnn_ae(net, G, nodes,
                                                      evaluate.DEVICE, seed,
                                                      A_norm_sp=A_norm_sp)
                with torch.no_grad():
                    s = model(A_sp, ae.latent(A_norm_sp)).cpu().numpy()
                scores.append(s)
                seed_taus.append(evaluate.kendall_taus(s, nodes, sir_list))
            if seed_stats is not None:
                seed_stats[net] = [float(np.mean(t)) for t in seed_taus]
            score = np.mean(scores, axis=0)
        else:
            A_norm, D = gnn_models.build_graph(G, nodes)
            A_norm, D = A_norm.to(evaluate.DEVICE), D.to(evaluate.DEVICE)
            with torch.no_grad():
                # the saved STGCN is a 15-run ensemble returning the mean
                score, _ = model(A_norm, D)
                score = score.squeeze(1).cpu().numpy()
        return evaluate.kendall_taus(score, nodes, sir_list)
    else:
        raise ValueError(name)
    scores = evaluate.model_scores(model, emb, nodes, C)
    return evaluate.kendall_taus(scores, nodes, sir_list)


def eval_centrality(name, net, G, set_, sir_list):
    """Kendall-tau list for a classical centrality index."""
    nodes = list(G.nodes())
    if name == 'BC':
        scores = evaluate.betweenness(net, G)
    elif name == 'DC':
        scores = dict(nx.degree_centrality(G))
    elif name == 'Vc':
        scores = Utils.Vc(G, set_.community(net))
    elif name == 'K-core':
        scores = dict(nx.core_number(G))
    elif name == 'ND':
        # ND including the node's own degree (matches the LP-RCNN ND channel)
        scores = Utils.neighbor_degree2(G)
    else:
        raise ValueError(name)
    scores = {node: scores[node] for node in nodes}
    return evaluate.kendall_taus([scores[node] for node in nodes], nodes, sir_list)


# --- Figures and table -----------------------------------------------------

def _plot_order(tau_dict, networks):
    """Methods ordered by their mean tau (LP-RCNN first), so the legend and
    the z-order follow the ranking and the figure stays readable."""
    means = {m: float(np.mean([np.mean(tau_dict[m][n]) for n in networks]))
             for m in METHODS}
    return sorted(METHODS, key=lambda m: -means[m])


def _save(fig, path):
    """Write the raster copy the manuscript embeds, plus a vector copy for the
    artwork upload.  Elsevier asks for at least 300 dpi on a raster figure and
    prefers vector artwork, so both are emitted from the same canvas."""
    fig.savefig(path, dpi=600)
    vec = os.path.splitext(path)[0] + '.pdf'
    fig.savefig(vec)
    print(f'saved: {path} (+ {vec})')


def plot_curves(tau_dict, networks):
    """Per-network tau vs beta/beta_th curves (appendix detail figure)."""
    order = _plot_order(tau_dict, networks)
    n_nets = len(networks)
    ncols = 4 if n_nets > 10 else 3
    nrows = int(np.ceil(n_nets / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, net in enumerate(networks):
        ax = axes[i]
        for method in order:
            ax.plot(BETA_RATIOS, tau_dict[method][net], label=method, **STYLE[method])
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
    fig.legend(handles, labels, loc='lower center', ncol=len(order),
               fontsize=9, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle('Node ranking capability: Kendall tau-b vs beta/beta_th',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    path = os.path.join(OUT_DIR, 'compare_tau_curves.png')
    _save(fig, path)
    plt.close(fig)
    print(f'saved: {path}')


def plot_average(tau_dict, networks):
    """Mean tau over the selected networks vs beta/beta_th (main figure)."""
    order = _plot_order(tau_dict, networks)
    fig, ax = plt.subplots(figsize=(8, 5.6))
    for method in order:
        avg = np.mean([tau_dict[method][net] for net in networks], axis=0)
        ax.plot(BETA_RATIOS, avg, label=method, **STYLE[method])
    ax.set_xlabel(r'$\beta/\beta_{th}$', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Kendall ' + r'$\tau$', fontsize=13, fontweight='bold')
    ax.set_title('Average Kendall tau-b over {} real-world networks'.format(len(networks)),
                 fontsize=13, fontweight='bold')
    ax.set_xticks([1.0, 1.3, 1.6, 2.0])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.round(np.arange(0.0, 1.01, 0.2), 1))
    ax.grid(alpha=0.3, ls='--')
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'compare_tau_average.png')
    _save(fig, path)
    plt.close(fig)
    print(f'saved: {path}')


def save_table_at_beta1_6(tau_dict, networks):
    """Table of tau at beta/beta_th = 1.6 for every method x network."""
    rows = {}
    for method in METHODS:
        rows[method] = {net: float(tau_dict[method][net][BETA_1_6_IDX]) for net in networks}
    df = pd.DataFrame(rows).T
    df.index.name = 'Method'
    df['Mean'] = df.mean(axis=1)
    df = df.sort_values('Mean', ascending=False)

    path = os.path.join(OUT_DIR, 'table_beta1_6.csv')
    df.to_csv(path)
    print('\n=== Kendall tau-b at beta/beta_th = 1.6 (* = best network) ===')
    evaluate.print_table(df)
    print(f'\nsaved: {path}')
    return df


# --- Main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='LP-RCNN vs baseline methods')
    ap.add_argument('--networks', type=str, default='',
                    help='comma-separated subset of networks to evaluate')
    ap.add_argument('--plot-only', action='store_true',
                    help='redraw the two figures from the saved '
                         'tau_curves_all.csv (no evaluation)')
    args = ap.parse_args()

    global OUT_DIR
    set_ = NetworkSet()
    if args.networks:
        sel = [s.strip() for s in args.networks.split(',') if s.strip()]
        networks = [n for n in set_.names() if n in sel]
        if networks != set_.names():
            OUT_DIR = os.path.join('figures', 'compare',
                                   'subset_' + '_'.join(networks))
            os.makedirs(OUT_DIR, exist_ok=True)
            print(f'[subset run] outputs redirected to {OUT_DIR} so the full '
                  f'benchmark tables are not overwritten')
    else:
        networks = set_.names()

    if args.plot_only:
        # redraw from the saved per-ratio table: no model is evaluated
        csv = os.path.join(OUT_DIR, 'tau_curves_all.csv')
        df = pd.read_csv(csv)
        tau_dict = {}
        for m in METHODS:
            sub_m = df[df['Method'] == m]
            d = {n: np.array(sub_m[sub_m['Network'] == n]
                             .sort_values('beta_ratio')['tau'].values,
                             dtype=float)
                 for n in set_.names()
                 if len(sub_m[sub_m['Network'] == n]) == len(BETA_RATIOS)}
            if d:
                tau_dict[m] = d
        networks = [n for n in set_.names()
                    if all(n in tau_dict[m] for m in tau_dict)]
        print(f'plot-only: redrawing {networks} from {csv}')
        plot_curves(tau_dict, networks)
        plot_average(tau_dict, networks)
        return

    print(f'compare_methods.py  device={evaluate.DEVICE}  networks={networks}')
    models = load_models()

    tau_dict = {m: {} for m in METHODS}
    seed_stats = {}
    for i, net in enumerate(networks):
        G, sir_list = set_.graph(net), set_.sir(net)
        print(f'[{i + 1}/{len(networks)}] {net} (N={G.number_of_nodes()})')
        for method in METHODS:
            if method in models:
                tau_dict[method][net] = eval_neural(models[method], method, net, G, set_, sir_list,
                                                    seed_stats=seed_stats)
            else:
                tau_dict[method][net] = eval_centrality(method, net, G, set_, sir_list)
        print('   done')

    # full per-ratio curves (appendix data + claim verification)
    curve_rows = [(m, n, float(b), float(tau_dict[m][n][i]))
                  for m in METHODS for n in networks
                  for i, b in enumerate(BETA_RATIOS)]
    cdf = pd.DataFrame(curve_rows, columns=['Method', 'Network', 'beta_ratio', 'tau'])
    cdf.to_csv(os.path.join(OUT_DIR, 'tau_curves_all.csv'), index=False)

    mean_df = evaluate.build_table(tau_dict, networks)
    print('\n=== Average Kendall tau-b over beta/beta_th = 1.0..2.0 ===')
    evaluate.print_table(mean_df)
    mean_df.to_csv(os.path.join(OUT_DIR, 'table_mean_tau.csv'))

    if seed_stats:
        # per-network tau of each AE init; the table averages the 5 scores
        sdf = pd.DataFrame({'Network': list(seed_stats.keys())})
        for s in range(5):
            sdf[f'seed{s}'] = [seed_stats[n][s] for n in sdf['Network']]
        sdf['Mean'] = sdf[[f'seed{s}' for s in range(5)]].mean(axis=1)
        sdf['Std'] = sdf[[f'seed{s}' for s in range(5)]].std(axis=1)
        sdf['Min'] = sdf[[f'seed{s}' for s in range(5)]].min(axis=1)
        sdf['Max'] = sdf[[f'seed{s}' for s in range(5)]].max(axis=1)
        spath = os.path.join(OUT_DIR, 'agnn_ae_seed_stats.csv')
        sdf.to_csv(spath, index=False)
        print(f'saved: {spath}')
        print(f'  per-network AE-seed std: mean {sdf["Std"].mean():.3f}, '
              f'min {sdf["Std"].min():.3f}, max {sdf["Std"].max():.3f}')
        print(f'  individual runs: min {sdf["Min"].min():.3f}, '
              f'max {sdf["Max"].max():.3f}')

    plot_curves(tau_dict, networks)
    plot_average(tau_dict, networks)
    save_table_at_beta1_6(tau_dict, networks)


if __name__ == '__main__':
    main()
