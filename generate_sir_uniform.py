# -*- coding: utf-8 -*-
"""Regenerate the SIR labels of the training networks under one uniform recipe.

Each ratio file comes from that graph's own edge list.  Recipe, as in ``generate_sir.py``: seed =
10 + i, beta = a * beta_th with beta_th = <k> / (<k^2> - <k>), miu = 1, N = 1000.

Parallelisation is over (graph, ratio) pairs only and nodes are visited in
``list(G.nodes())`` order, so the output matches the sequential run.  Both paths
need the same ``PYTHONHASHSEED`` (the shipped files used 0).

    python generate_sir_uniform.py --graphs BA3000_4
"""
import os
import sys
import time
import shutil
import argparse
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import networkx as nx
from multiprocessing import Pool, cpu_count

import Utils

# (name, graph file, output prefix of the 11 ratio files)
GRAPHS = [
    ('Train1000_4', './Networks/training/Train_1000_4.txt',
     './SIR results/Train_1000_4/BA_1000_4_'),
    ('BA2000_4', './Networks/training/BA_2000_4.txt',
     './SIR results/BA_2000_4/BA_2000_4_'),
    ('BA3000_4', './Networks/training/BA_3000_4.txt',
     './SIR results/BA_3000_4/BA_3000_4_'),
]

BETAS = [round(1.0 + 0.1 * i, 1) for i in range(11)]
RUNS = 1000
SEED = 10

_GRAPH_CACHE = {}


def _init(specs):
    """Pre-load every graph once per worker process."""
    global _GRAPH_CACHE
    for name, path, _ in specs:
        G = Utils.load_graph(path)
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        _GRAPH_CACHE[name] = G


def _run(task):
    """Simulate one (graph, ratio) pair.

    Parameters travel in ``task``: Windows spawns the workers, which would otherwise
    only see the module defaults.
    """
    name, i, prefix, runs = task
    G = _GRAPH_CACHE[name]
    a = BETAS[i]

    # --- identical to generate_sir.py -------------------------------------
    Utils.setup_seed(SEED + i)
    node_list = list(G.nodes())
    dc_list = np.array(list(dict(G.degree()).values()))
    beta = a * (float(dc_list.mean()) /
                (float((dc_list ** 2).mean()) - float(dc_list.mean())))
    vals = {node: Utils.SIR(G, infected=[node], beta=beta, miu=1, N=runs)
            for node in node_list}
    # ----------------------------------------------------------------------

    out = prefix + str(i) + '.csv'
    pd.DataFrame({'Node': list(vals.keys()),
                  'SIR': list(vals.values())}).to_csv(out, index=False)
    return dict(graph=name, ratio=a, out=out,
                n=len(vals), mean=float(np.mean(list(vals.values()))))


def backup(specs, ratios):
    """Copy the ratio files that are about to be overwritten."""
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    root = os.path.join('SIR results', f'_legacy_{stamp}')
    moved = 0
    for name, _, prefix in specs:
        for i in ratios:
            src = prefix + str(i) + '.csv'
            if not os.path.exists(src):
                continue
            dst_dir = os.path.join(root, name)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
            moved += 1
    if moved:
        print(f'备份 {moved} 个旧比率文件 -> {root}')
    else:
        print('没有需要备份的旧比率文件')
    return root


def main():
    ap = argparse.ArgumentParser(description='Uniform-recipe SIR regeneration')
    ap.add_argument('--graphs', type=str, default='',
                    help='comma-separated subset of ' +
                         ','.join(n for n, _, _ in GRAPHS))
    ap.add_argument('--ratios', type=str, default='',
                    help='comma-separated ratio indices 0..10 (default: all)')
    ap.add_argument('--runs', type=int, default=RUNS)
    ap.add_argument('--workers', type=int, default=0,
                    help='parallel workers (default: min(30, cpu_count-2))')
    ap.add_argument('--no-backup', action='store_true',
                    help='do not copy the old ratio files before overwriting')
    args = ap.parse_args()

    specs = GRAPHS
    if args.graphs:
        sel = [s.strip() for s in args.graphs.split(',') if s.strip()]
        specs = [s for s in GRAPHS if s[0] in sel]
        if not specs:
            sys.exit(f'--graphs 未匹配任何图，可选: {[n for n, _, _ in GRAPHS]}')

    ratios = ([int(x) for x in args.ratios.split(',') if x.strip()]
              if args.ratios else list(range(len(BETAS))))
    ratios = [r for r in ratios if 0 <= r < len(BETAS)]
    if not ratios:
        sys.exit('--ratios 必须是 0..%d 之间的索引' % (len(BETAS) - 1))

    tasks = [(name, i, prefix, args.runs)
             for name, _, prefix in specs for i in ratios]
    workers = args.workers or min(len(tasks), max(1, cpu_count() - 2))

    print(f'generate_sir_uniform.py  graphs={[n for n, _, _ in specs]}  '
          f'ratios={[BETAS[i] for i in ratios]}  runs={RUNS}  '
          f'tasks={len(tasks)}  workers={workers}')

    for name, path, prefix in specs:
        G = Utils.load_graph(path)
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        print(f'  {name:<12} N={G.number_of_nodes():>5} '
              f'E={G.number_of_edges():>6} -> {prefix}{{0..10}}.csv')

    if not args.no_backup:
        backup(specs, ratios)

    print('\n开始模拟（每个任务内部与 generate_sir.py 完全等价的随机流）...')
    t0 = time.perf_counter()
    results = []
    with Pool(workers, initializer=_init, initargs=(specs,)) as pool:
        for r in pool.imap_unordered(_run, tasks):
            results.append(r)
            print(f'  [{len(results):>2}/{len(tasks)}] {r["graph"]:<12} '
                  f'beta/beta_th={r["ratio"]:.1f}  mean SIR={r["mean"]:8.3f}  '
                  f'-> {r["out"]}', flush=True)

    dt = time.perf_counter() - t0
    print(f'\n完成 {len(results)} 个文件，用时 {dt / 60:.1f} min')


if __name__ == '__main__':
    main()
