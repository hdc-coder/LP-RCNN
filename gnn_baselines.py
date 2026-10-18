# -*- coding: utf-8 -*-
"""Standalone training driver for the AGNN / STGCN baselines.

The models are in ``gnn_models.py``.  Trains one on BA(1000,4), saves it to
``Models/`` and writes a per-model CSV; ``compare_methods.py`` loads the same
weights for the full table.

    python gnn_baselines.py --model agnn [--smoke]
"""
import os
import argparse
import warnings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import Utils
import evaluate
import gnn_models
from evaluate import NetworkSet, DEVICE

TRAIN_NET = './Networks/training/Train_1000_4.txt'
TRAIN_SIR_PREFIX = './SIR results/Train_1000_4/BA_1000_4_'
OUT_DIR = os.path.join('figures', 'baselines_gnn')
WEIGHTS = os.path.join('Models', '{}.pt')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs('Models', exist_ok=True)


def train(name, smoke=False):
    G = Utils.load_graph(TRAIN_NET)
    nodes = list(G.nodes())
    label = evaluate.train_label(TRAIN_SIR_PREFIX)
    y = torch.zeros(len(nodes), 1)
    for i, node in enumerate(nodes):
        y[i, 0] = label.get(node, 0.0)
    y = y.to(DEVICE).squeeze(1)

    if name == 'agnn':
        # official pipeline; the notebook seeds the BA side with 17, the
        # unified protocol with 10
        Utils.setup_seed(10)
        A_norm_sp = gnn_models.sparse_anorm(G, nodes).to(DEVICE)
        A_sp, D_ba = gnn_models.sparse_ai(G, nodes)
        A_sp, D_ba = A_sp.to(DEVICE), D_ba.to(DEVICE)
        ae_ba = gnn_models.AgnAE(len(nodes)).to(DEVICE)
        gnn_models.train_agnn_ae(ae_ba, A_norm_sp, D_ba,
                                 epochs=1 if smoke else 500)
        Z_ba = ae_ba.latent(A_norm_sp).detach()
        model = gnn_models.AgnRank().to(DEVICE)
        gnn_models.train_agnn_rank(model, A_sp, Z_ba, y,
                                   epochs=1 if smoke else 3000, verbose=True)
        return model, G, nodes, label

    # STGCN: 15 independently trained models, averaged at evaluation
    A_norm, D = gnn_models.build_graph(G, nodes)
    A_norm, D = A_norm.to(DEVICE), D.to(DEVICE)
    members = []
    n_seeds = 1 if smoke else 15
    for si, seed in enumerate(range(10, 10 + n_seeds)):
        Utils.setup_seed(seed)
        m = gnn_models.STGCN().to(DEVICE)
        gnn_models.train_stgcn(m, A_norm, D, y, ae_epochs=1 if smoke else 200,
                               rank_epochs=1 if smoke else 300)
        members.append(m)
        print(f'  STGCN member {si + 1}/{n_seeds} trained (seed {seed})',
              flush=True)
    model = gnn_models.Ensemble(members)
    return model, G, nodes, label


def evaluate_model(name, model, net_set, networks):
    rows, taus = {}, []
    seed_stats = {}
    for net in networks:
        G = net_set.graph(net)
        nodes = list(G.nodes())
        if name == 'agnn':
            # each network's own AE latent scored by the BA-trained rank
            # module, averaged over 5 AE inits (they can swing tau by ~0.6)
            A_norm_sp = gnn_models.sparse_anorm(G, nodes).to(DEVICE)
            A_sp, D = gnn_models.sparse_ai(G, nodes)
            A_sp = A_sp.to(DEVICE)
            model.eval()
            scores, seed_taus = [], []
            for seed in range(5):
                ae = gnn_models.load_or_train_agnn_ae(net, G, nodes, DEVICE,
                                                      seed, A_norm_sp=A_norm_sp)
                with torch.no_grad():
                    s = model(A_sp, ae.latent(A_norm_sp)).cpu().numpy()
                scores.append(s)
                seed_taus.append(evaluate.kendall_taus(s, nodes, net_set.sir(net)))
            score = np.mean(scores, axis=0)
            ratio_taus = evaluate.kendall_taus(score, nodes, net_set.sir(net))
            seed_stats[net] = [float(np.mean(t)) for t in seed_taus]
        else:
            A_norm, D = gnn_models.build_graph(G, nodes)
            A_norm, D = A_norm.to(DEVICE), D.to(DEVICE)
            model.eval()
            with torch.no_grad():
                # the saved STGCN is a 15-run ensemble returning the mean
                score, _ = model(A_norm, D)
                score = score.squeeze(1).cpu().numpy()
            ratio_taus = evaluate.kendall_taus(score, nodes, net_set.sir(net))
        mt = float(np.mean(ratio_taus))
        rows[net] = mt
        taus.append(mt)
        print(f'    {net:<10} tau = {mt:.4f}', flush=True)
    return rows, taus, seed_stats


def main():
    ap = argparse.ArgumentParser(description='AGNN / STGCN baseline training')
    ap.add_argument('--model', choices=['agnn', 'stgcn'], required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--networks', type=str, default='',
                    help='comma-separated subset of test networks')
    args = ap.parse_args()
    recipes = {'agnn': 'AE 500ep (lr0.01) + rank 3000ep (lr0.001, listMLE)',
               'stgcn': 'AE 200ep (lr0.01) + rank 300ep (lr1e-4, listMLE), 15-run ensemble'}

    print(f'gnn_baselines.py  model={args.model}  device={DEVICE}  '
          f'{"smoke" if args.smoke else recipes[args.model]}', flush=True)
    net_set = NetworkSet()
    networks = (net_set.names() if not args.networks
                else [n for n in net_set.names() if n in args.networks.split(',')])

    model, G, nodes, label = train(args.model, smoke=args.smoke)
    wpath = WEIGHTS.format(args.model)
    torch.save(model, wpath)
    print(f'saved weights: {wpath}', flush=True)

    rows, taus, seed_stats = evaluate_model(args.model, model, net_set, networks)

    df = pd.DataFrame({'Method': args.model.upper()}, index=[0])
    for net in networks:
        df[net] = rows[net]
    df['Mean'] = float(np.mean(taus))
    df['#Best'] = 0
    df = df[['Method'] + networks + ['Mean', '#Best']]
    path = os.path.join(OUT_DIR, f'table_{args.model}.csv')
    df.to_csv(path, index=False)
    print(f'\n=== {args.model.upper()} mean tau = {df["Mean"].iloc[0]:.4f} ===')
    print(f'saved: {path}')

    if seed_stats:
        # per-network, per-AE-seed tau (stability disclosure)
        sdf = pd.DataFrame({'Network': list(seed_stats.keys())})
        for s in range(5):
            sdf[f'seed{s}'] = [seed_stats[n][s] for n in sdf['Network']]
        sdf['Mean'] = sdf[[f'seed{s}' for s in range(5)]].mean(axis=1)
        sdf['Std'] = sdf[[f'seed{s}' for s in range(5)]].std(axis=1)
        sdf['Min'] = sdf[[f'seed{s}' for s in range(5)]].min(axis=1)
        sdf['Max'] = sdf[[f'seed{s}' for s in range(5)]].max(axis=1)
        spath = os.path.join(OUT_DIR, 'agnn_seed_stats.csv')
        sdf.to_csv(spath, index=False)
        print(f'saved: {spath}')
        print(f'  per-network AE-seed std: mean {sdf["Std"].mean():.3f}, '
              f'min {sdf["Std"].min():.3f}, max {sdf["Std"].max():.3f}')
        print(f'  individual runs: min {sdf["Min"].min():.3f}, '
              f'max {sdf["Max"].max():.3f}')


if __name__ == '__main__':
    main()
