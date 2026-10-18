# -*- coding: utf-8 -*-
"""Per-node neighborhood embeddings fed to the CNNs.

``L`` nodes per node: itself, then first-order neighbours, then higher-order ones
by BFS, highest degree first; encoded as a ``[C, L, L]`` matrix.  ``main`` =
degree -> RCNN, ``main1`` = ND/NCom/k-core -> M-RCNN, ``main_n_rcnn`` =
DC/HC/BC + contribution -> N-RCNN, ``main_ndrcnn`` = ND -> LP-RCNN.
"""
import numpy as np
import networkx as nx
import torch

import Utils


def _select_neighborhood(G, node, L):
    """``L-1`` neighbours of ``node``: highest degree first, then BFS fill.
    """
    subset = [node]
    # descending degree, ties by ascending node id (as in the BFS fill below)
    one_order = list(G.adj[node])
    selected_nei = sorted(
        one_order, key=lambda n: (-G.degree(n), str(n)))

    if len(one_order) >= L - 1:
        for nei in selected_nei:
            if (nei not in subset) and (len(subset) < L):
                subset.append(nei)
        return subset

    # top up with higher-order neighbours until L
    gap = (L - 1) - len(selected_nei)
    high_nei = set(selected_nei)
    neis = selected_nei
    count = 0
    while count < 50:
        new_order = set()
        for nei in neis:
            for each in G.adj[nei]:
                if (each != node) and (each not in high_nei):
                    new_order.add(each)
        # ``new_order`` is a set, so ``list()`` would iterate in hash-random
        # order (PYTHONHASHSEED) and make the selection irreproducible.
        new_order_list = sorted(new_order)
        degree_new = dict(G.degree(new_order_list))
        new_selected_nei = [i for i, j in sorted(degree_new.items(), key=lambda x: (-x[1], x[0]))]
        if len(new_selected_nei) >= gap:
            for i in range(gap):
                selected_nei.append(new_selected_nei[i])
            break
        elif len(new_selected_nei) < gap:
            for new in new_selected_nei:
                selected_nei.append(new)
                gap -= 1
            neis = new_order_list
            for each in neis:
                high_nei.add(each)
        count += 1

    for neii in selected_nei:
        if neii not in subset:
            subset.append(neii)
    return subset


def _pad(B, C, L):
    """Pad a ``[C, n, n]`` matrix to ``[C, L, L]`` (n <= L)."""
    if not isinstance(B, torch.Tensor):
        B = torch.from_numpy(np.asarray(B, dtype=np.float32))
    n = B.shape[-1]
    if n == L:
        return B.reshape(C, L, L)
    padded = torch.zeros(C, L, L)
    padded[:, :n, :n] = B
    return padded


def _build_embeddings(G, L, features, transform_fn, n_channels):
    """Embed every node with ``transform_fn``, then pad to ``[C, L, L]``.
    """
    data_dict = {}
    for node in G.nodes():
        subset = _select_neighborhood(G, node, L)
        if len(subset) == 1:                       # isolated node
            data_dict[node] = torch.zeros(n_channels, L, L)
            continue
        feat_lists = {name: [features[name][node]] for name in features}
        for nei in subset[1:]:
            for name in features:
                feat_lists[name].append(features[name][nei])
        _, node_A = Utils.generate_subgraph(G, subset)
        node_B = transform_fn(node_A, feat_lists)
        data_dict[node] = _pad(node_B, n_channels, L)
    return data_dict


# ----------------------------------------------------------------------------
# RCNN (single channel, degree)
# ----------------------------------------------------------------------------

def main(G, L):
    """Single-channel degree embedding -> input of the RCNN model."""
    features = {'deg': dict(G.degree())}

    def transform(A, feats):
        return Utils.transform(A, feats['deg'])

    return _build_embeddings(G, L, features, transform, 1)


# ----------------------------------------------------------------------------
# M-RCNN (three channels, ND / NCom / k-core)
# ----------------------------------------------------------------------------

def main1(G, L, community):
    """Three-channel ND / #communities / k-core embedding -> M-RCNN (Ou et al. 2022).

    Its ND is the traditional one: neighbours' degrees, own degree excluded.
    """
    features = {
        'nd': Utils.neighbor_degree(G),
        'com': dict(community),
        'ks': dict(nx.core_number(G)),
    }

    def transform(A, feats):
        return Utils.transform1(A, feats['nd'], feats['com'], feats['ks'])

    return _build_embeddings(G, L, features, transform, 3)


# ----------------------------------------------------------------------------
# N-RCNN (three channels, DC/HC/BC + neighbor contribution, adjacency base)
# ----------------------------------------------------------------------------

def _edge_bfs_subset(G, node, L):
    """``L-1`` edges by BFS, expanding the highest-degree frontier first (Luo & Lu 2023).

    Also returns each node's hop distance from the target.
    """
    subset = [node]
    dist = {node: 0}
    visited = {node}
    edges = []
    frontier = [node]
    level = 0
    while len(edges) < L - 1 and frontier:
        level += 1
        frontier = sorted(frontier, key=lambda n: G.degree(n), reverse=True)
        new_frontier = []
        for u in frontier:
            if len(edges) >= L - 1:
                break
            for v in sorted(G.neighbors(u), key=lambda n: G.degree(n), reverse=True):
                if len(edges) >= L - 1:
                    break
                if v not in visited:
                    edges.append((u, v))
                    visited.add(v)
                    subset.append(v)
                    dist[v] = level
                    new_frontier.append(v)
        frontier = new_frontier
    return subset, edges, dist


def main_n_rcnn(G, L, centralities=None):
    """N-RCNN embedding (Luo & Lu, ICPICS 2023).

    Degree / betweenness / H-index over the edge-BFS neighborhood, weighted by
    ``contribution(u_i) = CC(u_i)^2 - |d(u0, u_i) - 1|`` and padded to L x L.
    """
    if centralities is not None:
        dc, bc, hc, cc = centralities
    else:
        dc = dict(nx.degree_centrality(G))
        bc = dict(nx.betweenness_centrality(G))
        hc = Utils.calcHIndexValues(G)
        cc = dict(nx.closeness_centrality(G))

    data_dict = {}
    for node in G.nodes():
        subset, edges, dist = _edge_bfs_subset(G, node, L)
        if len(subset) == 1:                       # isolated node
            data_dict[node] = torch.zeros(3, L, L)
            continue
        enc = {n: i for i, n in enumerate(subset)}
        M = np.zeros((L, L))
        for a, b in edges:
            ia, ib = enc[a], enc[b]
            M[ia, ib] = 1
            M[ib, ia] = 1

        mats = []
        for feat in (dc, bc, hc):
            F = np.zeros((L, L))
            for i in range(L):
                for j in range(L):
                    if M[i, j] != 0 and i != j:
                        k = j if i < j else i
                        nk = subset[k]
                        contribution = cc[nk] ** 2 - abs(dist[nk] - 1)
                        F[i, j] = M[i, j] * feat[nk] * contribution
            mats.append(F)
        data_dict[node] = torch.FloatTensor(np.stack(mats, axis=0))
    return data_dict


# ----------------------------------------------------------------------------
# LP-RCNN (single channel, ND with the node's own degree included)
# ----------------------------------------------------------------------------

def main_ndrcnn(G, L):
    """Single-channel ND embedding -> input of LP-RCNN.

    ND(i) = k(i) + sum of the neighbours' degrees, so the node's own degree counts.
    """
    features = {'nd': Utils.neighbor_degree2(G)}

    def transform(A, feats):
        return Utils.transform2(A, feats['nd'])

    return _build_embeddings(G, L, features, transform, 1)
