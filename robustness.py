# -*- coding: utf-8 -*-
"""Sensitivity of the results to the seed and the training network.

Part A retrains on BA(1000,4) with five seeds; Part B retrains on BA(1000,4),
BA(2000,4) and BA(3000,4) with one.  Both label every training network the same
way: the mean over its eleven ratio files.  Embeddings and labels are
seed-independent, so they are built once per (kind, graph).
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import Utils
import Embeddings
import Models
import evaluate
from evaluate import NetworkSet, DEVICE, L

LR = 0.005
BATCH = 32
EPOCHS = 500
SEEDS = [10, 11, 12, 13, 14]

# (name, graph_path, ratio_prefix, cache_key).  ``ratio_prefix`` is the 11
# ratio files of that graph; every training network uses the mean over them.
# ``cache_key`` names the centrality cache of the *file*, not the display name:
# ``BA1000_4`` points at ``Train_1000_4.txt``, next to an unrelated
# ``BA_1000_4.txt``, and keying on the name made N-RCNN read the wrong graph.
TRAIN_NETS = [
    ('BA1000_4', './Networks/training/Train_1000_4.txt',
     './SIR results/Train_1000_4/BA_1000_4_', 'Train_1000_4'),
    ('BA2000_4', './Networks/training/BA_2000_4.txt',
     './SIR results/BA_2000_4/BA_2000_4_', 'BA2000_4'),
    ('BA3000_4', './Networks/training/BA_3000_4.txt',
     './SIR results/BA_3000_4/BA_3000_4_', 'BA3000_4'),
]

OUT_DIR = os.path.join('figures', 'robustness')
os.makedirs(OUT_DIR, exist_ok=True)

MULTISEED_CSV = os.path.join(OUT_DIR, 'multiseed_summary.csv')
PERNET_CSV = os.path.join(OUT_DIR, 'multiseed_per_network.csv')
MULTINET_CSV = os.path.join(OUT_DIR, 'multinetwork_summary.csv')


def _append(df, path):
    """Append ``df`` to ``path``, creating it with a header on first write."""
    header = not os.path.exists(path)
    df.to_csv(path, mode='a', header=header, index=False)


def _load_done(path):
    """Set of ``(model, train_net, seed)`` rows already written to ``path``."""
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    if len(df) == 0:
        return set()
    return set(df[['model', 'train_net', 'seed']].apply(tuple, axis=1))


# ---------------------------------------------------------------------------
# Model / embedding definitions (shared with train.py / time_to_accuracy.py)
# ---------------------------------------------------------------------------

def build_models():
    """``{model: (builder, embedding_kind, C, use_listmle, batch)}``; batch and lr
    match ``compare_methods.py`` so seed 10 reproduces Table 1.
    """
    return {
        'RCNN':   (lambda: Models.CNN(L),              'degree', 1, False, 32),
        'M-RCNN': (lambda: Models.CNN1(L),             'mrcnn',  3, False, 32),
        'N-RCNN': (lambda: Models.N_RCNN(L),           'nrcnn',  3, False, 32),
        'LP-RCNN':(lambda: Models.DC_RCNN_MS_DIL(L, in_channels=1), 'nd', 1, True, 32),
    }


def make_emb(kind, G, net_name=''):
    """Embedding dict of ``G`` for the given embedding kind."""
    if kind == 'degree':
        return Embeddings.main(G, L)
    if kind == 'nd':
        return Embeddings.main_ndrcnn(G, L)
    if kind == 'mrcnn':
        _, com, _ = Utils.Louvain(G)
        return Embeddings.main1(G, L, com)
    if kind == 'nrcnn':
        dc = {n: G.degree(n) / (G.number_of_nodes() - 1) for n in G.nodes()}
        bc = evaluate.betweenness(net_name, G)
        hc = Utils.calcHIndexValues(G)
        cc = evaluate.closeness(net_name, G)
        return Embeddings.main_n_rcnn(G, L, centralities=(dc, bc, hc, cc))
    raise ValueError(kind)


def stack(emb, C):
    nodes = list(emb.keys())
    X = torch.zeros(len(nodes), C, L, L)
    for i, node in enumerate(nodes):
        m = emb[node]
        X[i] = m if isinstance(m, torch.Tensor) else torch.from_numpy(
            np.asarray(m, dtype=np.float32))
    return X, nodes


def build_loader(emb, label, C, batch=BATCH):
    X, nodes = stack(emb, C)
    y = torch.zeros(len(nodes), 1)
    for i, node in enumerate(nodes):
        y[i, 0] = label.get(node, 0.0)
    return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=True)


# ---------------------------------------------------------------------------
# Training (fresh weights per seed)
# ---------------------------------------------------------------------------

def train_model(model, loader, use_listmle, lr=LR, epochs=EPOCHS):
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for data, tgt in loader:
            data = data.to(DEVICE)
            tgt = tgt.float().to(DEVICE)
            opt.zero_grad()
            pred = model(data)
            loss = (Utils.listmle_loss(pred, tgt) if use_listmle
                    else nn.MSELoss()(pred, tgt))
            loss.backward()
            opt.step()
    return model


# ---------------------------------------------------------------------------
# Evaluation on the 9 real-world networks (mean tau over beta, then networks)
# ---------------------------------------------------------------------------

def eval_on_real(model, emb_cache, emb_kind, C, net_set, networks):
    """Mean Kendall tau-b over the 9 networks (each averaged over beta)."""
    taus = []
    model = model.to(DEVICE).eval()
    for net in networks:
        G, sir_list = net_set.graph(net), net_set.sir(net)
        key = (emb_kind, net)
        if key not in emb_cache:
            emb_cache[key] = make_emb(emb_kind, G, net)
        emb = emb_cache[key]
        nodes = list(emb.keys())
        scores = evaluate.model_scores(model, emb, nodes, C)
        taus.append(float(np.mean(evaluate.kendall_taus(scores, nodes, sir_list))))
    return float(np.mean(taus)), {net: t for net, t in zip(networks, taus)}


# ---------------------------------------------------------------------------
# Training-network labels (mean SIR over the 11 infection rates)
# ---------------------------------------------------------------------------

def train_labels(sir_prefix):
    """``evaluate.train_label`` for a training network, so both use one rule.
    """
    return evaluate.train_label(sir_prefix)


# ---------------------------------------------------------------------------
# Figure: how far the ranking moves when the seed changes
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='LP-RCNN robustness (seeds / training nets)')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--seeds', type=str, default='')
    ap.add_argument('--train-nets', type=str, default='')
    ap.add_argument('--models', type=str, default='',
                    help='comma-separated subset of models, e.g. RCNN,M-RCNN')
    ap.add_argument('--skip-multiseed', action='store_true',
                    help='skip Part A (multi-seed on BA(1000,4))')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    args = ap.parse_args()
    epochs = args.epochs

    if args.smoke:
        seeds, net_names = [10], ['BA1000_4', 'BA2000_4']
    else:
        seeds = [int(s) for s in args.seeds.split(',') if s] or SEEDS
        net_names = ([s.strip() for s in args.train_nets.split(',') if s.strip()]
                     or [n for n, _, _ in TRAIN_NETS])
    nets_by_name = {n: (path, prefix, ckey)
                    for n, path, prefix, ckey in TRAIN_NETS}
    paper_net = 'BA1000_4'

    print(f'robustness.py  device={DEVICE}  seeds={seeds}  '
          f'train_nets={net_names}  epochs={epochs}', flush=True)
    net_set = NetworkSet()
    networks = net_set.names()
    models = build_models()
    if args.models:
        sel = [s.strip() for s in args.models.split(',') if s.strip()]
        models = {k: v for k, v in models.items() if k in sel}

    emb_cache = {}          # (emb_kind, graph_name) -> embedding dict
    done_seed = _load_done(MULTISEED_CSV)   # resumable: (model, train_net, seed)
    done_net = _load_done(MULTINET_CSV)

    # --- Part A: multi-seed on the reference training network BA(1000,4) ---
    if paper_net in net_names and not args.skip_multiseed:
        path, sir_prefix, cache_key = nets_by_name[paper_net]
        G = Utils.load_graph(path)
        label = train_labels(sir_prefix)
        print(f'\n### Part A: multi-seed on {paper_net} '
              f'(N={G.number_of_nodes()}) ###')

        train_data = {}
        for mname, (_, kind, C, use_listmle, batch) in models.items():
            key = (kind, paper_net)
            if key not in emb_cache:
                emb_cache[key] = make_emb(kind, G, cache_key)
            train_data[mname] = (build_loader(emb_cache[key], label, C,
                                              batch=batch), C, use_listmle)

        for seed in seeds:
            Utils.setup_seed(seed)
            print(f'\n  --- seed {seed} ---')
            for mname, (builder, kind, C, use_listmle, batch) in models.items():
                if (mname, paper_net, seed) in done_seed:
                    print(f'    {mname:<9} skip (already done)')
                    continue
                loader, C, use_listmle = train_data[mname]
                model = train_model(builder(), loader, use_listmle, epochs=epochs,
                                    lr=LR if mname == 'LP-RCNN' else 0.001)
                mean_tau, per_net = eval_on_real(model, emb_cache, kind, C,
                                                 net_set, networks)
                _append(pd.DataFrame(
                    [dict(model=mname, train_net=paper_net, seed=seed,
                          mean_tau=mean_tau)]), MULTISEED_CSV)
                _append(pd.DataFrame(
                    [dict(model=mname, train_net=paper_net, seed=seed,
                          network=net, tau=per_net[net]) for net in networks]),
                    PERNET_CSV)
                print(f'    {mname:<9} mean-tau = {mean_tau:.4f}', flush=True)

    # --- Part B: cross training-network (single seed) -----------------------
    seed = seeds[0]
    print(f'\n### Part B: multi training-network (seed={seed}, unified labels) ###')
    for tname in net_names:
        path, sir_prefix, cache_key = nets_by_name[tname]
        G = Utils.load_graph(path)
        label = train_labels(sir_prefix)
        print(f'\n  training network: {tname}  (N={G.number_of_nodes()})')
        Utils.setup_seed(seed)
        for mname, (builder, kind, C, use_listmle, batch) in models.items():
            if (mname, tname, seed) in done_net:
                print(f'    {mname:<9} skip (already done)')
                continue
            key = (kind, tname)
            if key not in emb_cache:
                emb_cache[key] = make_emb(kind, G, cache_key)
            loader = build_loader(emb_cache[key], label, C, batch=batch)
            model = train_model(builder(), loader, use_listmle, epochs=epochs,
                                lr=LR if mname == 'LP-RCNN' else 0.001)
            mean_tau, _ = eval_on_real(model, emb_cache, kind, C, net_set, networks)
            _append(pd.DataFrame(
                [dict(model=mname, train_net=tname, seed=seed, mean_tau=mean_tau)]),
                MULTINET_CSV)
            print(f'    {mname:<9} mean-tau = {mean_tau:.4f}', flush=True)

    # --- summary (read back from the CSVs, so a resumed run aggregates all) -
    if os.path.exists(MULTINET_CSV):
        df_mn = pd.read_csv(MULTINET_CSV)
        print('\n=== multi training-network (seed=%d) ===' % seed)
        pivot = df_mn.pivot(index='train_net', columns='model', values='mean_tau')
        print(pivot.round(4).to_string())
        print(f'saved: {MULTINET_CSV}')

    if os.path.exists(MULTISEED_CSV):
        df_ms = pd.read_csv(MULTISEED_CSV)
        print('\n=== multi-seed summary (mean +/- std over seeds) ===')
        grp = df_ms.groupby('model')['mean_tau'].agg(['mean', 'std', 'min', 'max'])
        grp = grp.rename(columns={'mean': 'mean_tau', 'std': 'std'})
        grp['mean+std'] = grp['mean_tau'] + grp['std']
        grp['mean-std'] = grp['mean_tau'] - grp['std']
        print(grp.round(4).to_string())
        print(f'saved: {MULTISEED_CSV}')
        print(f'saved: {PERNET_CSV}')


if __name__ == '__main__':
    main()
