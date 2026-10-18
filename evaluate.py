# -*- coding: utf-8 -*-
"""Shared dataset loading, evaluation and plotting helpers.

The nine networks, their SIR labels and the Louvain partition are loaded once,
in one way.  Protocol: train on BA(1000,4) with L = 28, rank the nine networks,
score by Kendall tau-b over eleven ratios beta/beta_th = 1.0 ... 2.0.
"""
import os
import hashlib
import json
import numpy as np
import pandas as pd
import networkx as nx
import torch
from scipy import stats

import Utils

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

L = 28                                        # neighborhood-network size
BETA_RATIOS = np.arange(1.0, 2.1, 0.1)        # beta/beta_th values (11 bins)
BETA_1_6_IDX = int(np.where(np.isclose(BETA_RATIOS, 1.6))[0][0])   # = 6
BETA_2_0_IDX = int(np.where(np.isclose(BETA_RATIOS, 2.0))[0][0])   # = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# (display name, network file, SIR label prefix)
NETWORKS = [
    ('Facebook',  './Networks/real/facebook_combined.txt', './SIR results/Facebook/Facebook_'),
    ('Faa',       './Networks/real/faa.txt',               './SIR results/Faa/Faa_'),
    ('Ham',       './Networks/real/Peh_edge.txt',          './SIR results/Ham/Ham_'),
    ('Hep',       './Networks/real/CA-HepTh.txt',          './SIR results/Hep/Hep_'),
    ('LastFM',    './Networks/real/LastFM.txt',            './SIR results/LastFM/LastFM_'),
    ('Vidal',     './Networks/real/vidal.txt',             './SIR results/vidal/vidal_'),
    ('Email',     './Networks/real/email.txt',             './SIR results/Email/Email_'),
    ('PowerGrid', './Networks/real/powergrid.txt',         './SIR results/powergrid/powergrid_'),
    ('NS',        './Networks/real/NS.txt',                './SIR results/NS/NS_'),
]

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Training labels -- one protocol for every network, synthetic or real
# ---------------------------------------------------------------------------

def train_label(sir_prefix, n=None):
    """``{node: SIR}`` for a training network: mean influence over the eleven ratios.
    """
    if n is None:
        n = len(BETA_RATIOS)
    sir_list = Utils.load_sir_list(sir_prefix, n=n)
    nodes = set().union(*[set(s.keys()) for s in sir_list])
    return {k: float(np.mean([s[k] for s in sir_list])) for k in nodes}


def label_files(sir_prefix, n=None):
    """The ``n`` per-ratio label files that ``train_label`` averages over."""
    if n is None:
        n = len(BETA_RATIOS)
    return [sir_prefix + str(i) + '.csv' for i in range(n)]


# ---------------------------------------------------------------------------
# Dataset loading (cached)
# ---------------------------------------------------------------------------

class NetworkSet:
    """Loads and caches graphs, SIR labels and Louvain communities."""

    def __init__(self, networks=None):
        self.networks = networks or NETWORKS
        self._meta = {name: (path, sir) for name, path, sir in self.networks}
        self._graphs = {}
        self._sirs = {}
        self._communities = {}

    def names(self):
        return [name for name, _, _ in self.networks]

    def graph(self, name):
        """NetworkX graph without self-loops."""
        if name not in self._graphs:
            G = Utils.load_graph(self._meta[name][0])
            G.remove_edges_from(nx.selfloop_edges(G))
            self._graphs[name] = G
        return self._graphs[name]

    def sir(self, name):
        """List of ``{node: SIR}`` dicts, one per beta/beta_th ratio."""
        if name not in self._sirs:
            self._sirs[name] = Utils.load_sir_list(self._meta[name][1],
                                                  n=len(BETA_RATIOS))
        return self._sirs[name]

    def community(self, name):
        """Louvain community id per node (needed for M-RCNN and the Vc index)."""
        if name not in self._communities:
            _, com_num, _ = Utils.Louvain(self.graph(name))
            self._communities[name] = com_num
        return self._communities[name]


def _graph_sha256(G):
    """Canonical SHA-256 of a graph, independent of the file's edge order."""
    h = hashlib.sha256()
    h.update(str(G.number_of_nodes()).encode('utf-8'))
    for node in sorted((str(n) for n in G.nodes())):
        h.update(f'	{node}	{G.degree(node)}'.encode('utf-8'))
    for u, v in sorted(tuple(sorted((str(u), str(v)))) for u, v in G.edges()):
        h.update(f'	{u}	{v}'.encode('utf-8'))
    return h.hexdigest()


def cached_centrality(name, G, key, fn):
    """Per-node score dict from ``cache/``, or computed and stored.

    The sidecar holds the graph SHA-256, so an edited network file yields a new
    entry.
    """
    path = os.path.join(CACHE_DIR, f'{name}_{key}.json')
    meta_path = path + '.meta.json'
    fingerprint = _graph_sha256(G)
    if os.path.exists(path) and os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        if meta.get('graph_sha256') == fingerprint:
            with open(path, 'r', encoding='utf-8') as f:
                return {str(k): float(v) for k, v in json.load(f).items()}
    print(f'  [cache] computing {key} for {name} ...')
    values = fn(G)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({str(k): float(v) for k, v in values.items()}, f)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({'graph_sha256': fingerprint}, f)
    return values


def betweenness(name, G):
    """Betweenness, cached.  Parallel exact Brandes, matching networkx up to rounding.
    """
    return cached_centrality(name, G, 'bc', betweenness_parallel)


def closeness(name, G):
    """Closeness centrality, cached to disk (used by the N-RCNN embedding)."""
    return cached_centrality(name, G, 'cc', nx.closeness_centrality)


# --- parallel exact betweenness centrality ---------------------------------
# Brandes (2001) computes each source node's dependency independently, so the
# per-source contributions add up exactly across workers.

_BC_ADJ = None            # global adjacency list shared by the workers


def _bc_worker_init(adj):
    global _BC_ADJ
    _BC_ADJ = adj


def _bc_chunk(sources):
    """Brandes dependency over a chunk of sources, counted over ordered pairs.
    """
    from collections import deque
    global _BC_ADJ
    adj = _BC_ADJ
    betweenness = {}
    for s in sources:
        sigma = dict.fromkeys(adj, 0.0)
        sigma[s] = 1.0
        dist = {s: 0}
        pred = {s: []}
        order = []
        queue = deque([s])
        while queue:
            v = queue.popleft()
            order.append(v)
            for w in adj[v]:
                if w not in dist:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                    pred[w] = [v]
                    sigma[w] += sigma[v]
                elif dist[w] == dist[v] + 1:
                    pred[w].append(v)
                    sigma[w] += sigma[v]
        delta = dict.fromkeys(adj, 0.0)
        while order:
            w = order.pop()
            coef = (1.0 + delta[w]) / sigma[w]
            for v in pred[w]:
                delta[v] += sigma[v] * coef
            if w != s:
                betweenness[w] = betweenness.get(w, 0.0) + delta[w]
    return betweenness


def betweenness_parallel(G, n_workers=None):
    """Exact betweenness centrality of an undirected graph (parallel Brandes)."""
    import multiprocessing as mp

    nodes = list(G.nodes())
    n = len(nodes)
    if n < 2000:                                    # small graph: single process
        return nx.betweenness_centrality(G)
    if n_workers is None:
        n_workers = min(16, max(1, os.cpu_count() - 2))

    # adjacency list in node order (fast, picklable)
    adj = {v: list(G.neighbors(v)) for v in nodes}
    chunks = np.array_split(np.asarray(nodes, dtype=object), n_workers)
    chunks = [list(c) for c in chunks if len(c) > 0]

    try:
        with mp.Pool(n_workers, initializer=_bc_worker_init, initargs=(adj,)) as pool:
            partials = pool.map(_bc_chunk, chunks)
    except Exception:
        # fall back to the networkx implementation if multiprocessing fails
        return nx.betweenness_centrality(G)

    total = {}
    for part in partials:
        for v, val in part.items():
            total[v] = total.get(v, 0.0) + val
    # our accumulation counts ordered pairs; networkx normalizes undirected BC
    # by 2/((n-1)(n-2)), so the combined factor for the ordered sum is 1/((n-1)(n-2))
    scale = 1.0 / ((n - 1) * (n - 2))
    return {v: total.get(v, 0.0) * scale for v in nodes}


# ---------------------------------------------------------------------------
# Ranking / evaluation helpers
# ---------------------------------------------------------------------------

def to_tensor(emb, nodes, C, L=28):
    """Pack a dict of per-node embeddings into one ``[N, C, L, L]`` tensor."""
    X = torch.zeros(len(nodes), C, L, L)
    for i, node in enumerate(nodes):
        m = emb[node]
        X[i] = m if isinstance(m, torch.Tensor) else torch.from_numpy(np.asarray(m, dtype=np.float32))
    return X


@torch.no_grad()
def model_scores(model, emb, nodes, C, L=28):
    """Forward the model over all nodes; ``L`` is the embedding size.
    """
    model = model.to(DEVICE).eval()
    X = to_tensor(emb, nodes, C, L).to(DEVICE)
    return model(X).cpu().numpy().flatten()


def kendall_taus(scores, nodes, sir_list):
    """Kendall tau-b against each SIR ground truth.

    Raw vectors go to ``scipy.stats.kendalltau``, so ties get the tau-b correction
    and no tie-breaking is applied beforehand.
    """
    if isinstance(scores, dict):
        scores = [scores[node] for node in nodes]
    scores = np.asarray(scores, dtype=float)
    taus = []
    for sir in sir_list:
        target = np.asarray([sir[node] for node in nodes], dtype=float)
        # tau-b is undefined on a constant vector (e.g. an AGNN head that
        # outputs zero for every node); such a model has no ranking ability,
        # so the conventional 0.0 is reported.
        if np.unique(scores).size < 2 or np.unique(target).size < 2:
            taus.append(0.0)
        else:
            taus.append(stats.kendalltau(scores, target)[0])
    return taus


def evaluate_scores(G, nodes, sir_list, scores):
    """Wrapper around ``kendall_taus`` accepting a ``{node: score}`` dict
    (used for the traditional centrality methods and trained models alike)."""
    return kendall_taus([scores[node] for node in nodes], nodes, sir_list)


# ---------------------------------------------------------------------------
# Table / figure helpers
# ---------------------------------------------------------------------------

def build_table(tau_dict, networks=None):
    """``{method: {network: taus}}`` -> DataFrame; ``Mean`` averages the ratios.
    """
    networks = networks or [n for n, _, _ in NETWORKS]
    rows = {}
    for method, by_net in tau_dict.items():
        rows[method] = {net: float(np.mean(by_net[net])) for net in networks}
    df = pd.DataFrame(rows).T
    df.index.name = 'Method'
    df['Mean'] = df.mean(axis=1)
    df['#Best'] = 0
    for net in networks:
        best = df[net].idxmax()
        df.loc[best, '#Best'] += 1
    return df.sort_values('Mean', ascending=False)


def print_table(df, precision=3):
    """Pretty-print a summary table, marking the per-network best value."""
    header = f"{df.index.name:<16}" + ''.join(f"{c:>10}" for c in df.columns)
    print(header)
    print('-' * len(header))
    for method, row in df.iterrows():
        line = f"{method:<16}"
        for col in df.columns:
            v = row[col]
            if col in ('Mean', '#Best'):
                line += f"{float(v):>10.{precision}f}" if col == 'Mean' else f"{int(v):>10}"
            else:
                star = '*' if v == df[col].max() else ' '
                line += f"{v:>9.{precision}f}{star}"
        print(line)
    print('-' * len(header))
