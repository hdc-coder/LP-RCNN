# -*- coding: utf-8 -*-
"""Generate SIR ground-truth labels for one network.

Per node the process runs ``--runs`` times at each ratio and the mean final
outbreak size goes to ``{out_prefix}{i}.csv``; ``--combined`` adds the mean over
all ratios.  The seed resets for every ratio.

``Utils.SIR`` walks a set of node-id strings, so the values also depend on
``PYTHONHASHSEED``.  Pin it to reproduce the shipped files:

    PYTHONHASHSEED=0 python generate_sir.py --network ... --out ... --seed 10
"""
import os
import argparse
import warnings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import networkx as nx

import Utils


def main():
    ap = argparse.ArgumentParser(description='Generate SIR label files')
    ap.add_argument('--network', required=True, help='edge-list graph file')
    ap.add_argument('--out', required=True,
                    help='output prefix; ratio i is written to {out}{i}.csv')
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--runs', type=int, default=1000)
    ap.add_argument('--betas', type=str,
                    default='1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0',
                    help='comma-separated beta/beta_th ratios')
    ap.add_argument('--combined', action='store_true',
                    help='also write {out}combined.csv (mean over all ratios)')
    args = ap.parse_args()

    betas = [float(x) for x in args.betas.split(',') if x.strip()]
    G = Utils.load_graph(args.network)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    print(f'generate_sir.py  N={G.number_of_nodes()}  E={G.number_of_edges()}  '
          f'betas={betas}  runs={args.runs}  seed={args.seed}')

    def sir_dict_runs(graph, beta=0.1, miu=1, real_beta=None, a=1.5):
        node_list = list(graph.nodes())
        dc_list = np.array(list(dict(graph.degree()).values()))
        beta = a * (float(dc_list.mean()) /
                    (float((dc_list ** 2).mean()) - float(dc_list.mean())))
        result = {}
        for node in node_list:
            result[node] = Utils.SIR(graph, infected=[node], beta=beta,
                                     miu=miu, N=args.runs)
        return result

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    frames = []
    for i, a in enumerate(betas):
        Utils.setup_seed(args.seed + i)
        vals = sir_dict_runs(G, real_beta=True, a=a)
        Utils.save_sir_dict(vals, args.out + str(i) + '.csv')
        frames.append(pd.DataFrame({'Node': list(vals.keys()),
                                    'SIR': list(vals.values())}))
        print(f'saved: {args.out}{i}.csv  beta/beta_th={a}')

    if args.combined:
        merged = pd.concat([f.set_index('Node') for f in frames], axis=1)
        combined = pd.DataFrame({'Node': merged.index,
                                 'SIR': merged.mean(axis=1).values})
        combined.to_csv(args.out + 'combined.csv', index=False)
        print(f'saved: {args.out}combined.csv')


if __name__ == '__main__':
    main()
