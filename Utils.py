# -*- coding: utf-8 -*-
"""Core utilities for LP-RCNN.

Graph I/O, importance indices, embedding transforms, loaders, the listMLE loss,
the training loops and the SIR simulator.
"""
import os
import hashlib
import random
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import community.community_louvain as community


# ----------------------------------------------------------------------------
# Environment / reproducibility
# ----------------------------------------------------------------------------

def setup_seed(seed):
    """Seed torch / numpy / CUDA and make cuDNN deterministic.

    ``benchmark`` off and ``deterministic`` on, or GPU results drift between runs.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
# Graph I/O
# ----------------------------------------------------------------------------

def load_graph(path):
    """Read an undirected network from an edge list file."""
    return nx.read_edgelist(path, create_using=nx.Graph())


def file_sha256(path, chunk_size=1 << 20):
    """SHA-256 of a file, used for dataset/model provenance sidecars."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_sir_list(path, n=10):
    """Load ``n`` SIR files ``{path}{i}.csv``; one ``{node: SIR}`` dict per ratio.

    Each file is the mean final outbreak size at ratio 1.0 + 0.1 i.
    """
    sir_list = []
    for i in range(n):
        sir = pd.read_csv(path + str(i) + '.csv')
        sir_list.append(dict(zip(np.array(sir['Node'], dtype=str), sir['SIR'])))
    return sir_list


# ----------------------------------------------------------------------------
# Node importance indices
# ----------------------------------------------------------------------------

def neighbor_degree(G):
    """Traditional ND: sum of the degrees of a node's neighbours (Lü et al. 2016)."""
    degree = dict(G.degree())
    n_degree = {}
    for node in G.nodes():
        n_degree[node] = sum(degree[nei] for nei in G.adj[node])
    return n_degree


def neighbor_degree2(G):
    """ND with the node's own degree: k(i) + sum of neighbour degrees (LP-RCNN).
    """
    degree = dict(G.degree())
    n_degree = {}
    for node in G.nodes():
        n_degree[node] = degree[node] + sum(degree[nei] for nei in G.adj[node])
    return n_degree


def count_h_index(h_list):
    """Largest i such that at least i entries of h_list are >= i."""
    h_list = sorted(h_list, reverse=True)
    i = 0
    while i < len(h_list) and h_list[i] > i:
        i += 1
    return i


def _cal_h_index(G, n, h_neg_dic):
    """Recursively compute the n-th order H-index of every node."""
    assert n >= 0, 'n>=0'
    if n == 0:
        return {n_i: G.degree(n_i) for n_i in G.nodes()}
    n -= 1
    prev = _cal_h_index(G, n, h_neg_dic)
    return {n_i: count_h_index([prev[neg] for neg in h_neg_dic[n_i]]) for n_i in G.nodes()}


def calcHIndexValues(G, n=1):
    """n-th order H-index of every node (Lü et al., Nat. Commun. 7 (2016) 10168).
    """
    h_neg_dic = {n_i: list(G.neighbors(n_i)) for n_i in G.nodes()}
    result = _cal_h_index(G, n, h_neg_dic)
    return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))


def Vc(G, community_dic):
    """Vc index: number of communities a node (and its neighbors) spans."""
    vc = {}
    for node in G.nodes():
        com_set = {community_dic[node]}
        for nei in G.adj[node]:
            com_set.add(community_dic[nei])
        vc[node] = len(com_set)
    return vc


def Louvain(G):
    """Louvain partition with a fixed ``random_state``.

    Returns ``{com_id: [nodes]}``, ``{node: #communities spanned}`` and ``{node:
    size of its own community}``.
    """
    partition = community.best_partition(G, random_state=42)
    community_name = set(partition.values())
    community_dic = {each: [] for each in community_name}
    for node, cid in partition.items():
        community_dic[cid].append(node)

    com_num, com_size = {}, {}
    for node in G.nodes():
        com_size[node] = len(community_dic[partition[node]])
        com_set = set([partition[node]])
        for nei in G.adj[node]:
            com_set.add(partition[nei])
        com_num[node] = len(com_set)
    return community_dic, com_num, com_size


# ----------------------------------------------------------------------------
# Subgraph extraction and embedding transforms
# ----------------------------------------------------------------------------

def generate_subgraph(G, node_list):
    """Build the (symmetric) adjacency matrix of the subgraph induced by ``node_list``."""
    L = len(node_list)
    encode = dict(zip(node_list, range(L)))
    subgraph = nx.subgraph(G, node_list)
    A = np.zeros([L, L])
    for i, j in subgraph.edges():
        if i != j:
            A[encode[i], encode[j]] = 1
            A[encode[j], encode[i]] = 1
    return nx.Graph(), A


def transform(A, degree_list):
    """Degree-weighted adjacency; only row/column 0 (the target) is weighted.
    """
    B = A.copy()
    d = np.array(degree_list)
    B[0, 1:] = A[0, 1:] * d[1:]
    B[1:, 0] = A[1:, 0] * d[1:]
    for i in range(len(d)):
        B[i, i] = d[i]
    return B


def transform1(A, nd_list, com_list, shell_list):
    """Three-channel M-RCNN embedding: ND / NCom / k-core weighted adjacency."""
    B1, B2, B3 = A.copy(), A.copy(), A.copy()
    nd, com, ks = np.array(nd_list), np.array(com_list), np.array(shell_list)

    B1[0, 1:] = B1[0, 1:] * nd[1:]
    B1[1:, 0] = B1[1:, 0] * nd[1:]
    B2[0, 1:] = B2[0, 1:] * com[1:]
    B2[1:, 0] = B2[1:, 0] * com[1:]
    B3[0, 1:] = B3[0, 1:] * ks[1:]
    B3[1:, 0] = B3[1:, 0] * ks[1:]

    for i in range(len(nd)):
        B1[i, i] = nd[i]
        B2[i, i] = com[i]
        B3[i, i] = ks[i]

    B = torch.zeros(3, A.shape[0], A.shape[0])
    B[0] = torch.from_numpy(B1).float()
    B[1] = torch.from_numpy(B2).float()
    B[2] = torch.from_numpy(B3).float()
    return B


def transform2(A, nd_list):
    """Single-channel ND embedding (used by LP-RCNN)."""
    B = A.copy()
    d = np.array(nd_list)
    B[0, 1:] = B[0, 1:] * d[1:]
    B[1:, 0] = B[1:, 0] * d[1:]
    for i in range(len(d)):
        B[i, i] = d[i]
    return B


# ----------------------------------------------------------------------------
# DataLoaders
# ----------------------------------------------------------------------------

def _make_loader(data, label, batch_size, L, n_channels):
    """Align X and y by node key, so the two cannot drift apart.
    """
    common_nodes = [node for node in data if node in label]
    if len(common_nodes) != len(data) or len(common_nodes) != len(label):
        raise ValueError(
            f'embedding/label node mismatch: {len(data)} embeddings, '
            f'{len(label)} labels, {len(common_nodes)} in common')
    X = torch.empty(len(common_nodes), n_channels, L, L)
    y = torch.empty(len(common_nodes), 1)
    for inx, node in enumerate(common_nodes):
        matrix = data[node]
        X[inx] = torch.from_numpy(matrix) if not isinstance(matrix, torch.Tensor) else matrix
        y[inx, 0] = label[node]
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)


def Get_DataLoader(data, label, batch_size, L):
    """DataLoader for single-channel embeddings ``[N, 1, L, L]``."""
    return _make_loader(data, label, batch_size, L, 1)


def Get_DataLoader1(data, label, batch_size, L):
    """DataLoader for three-channel embeddings ``[N, 3, L, L]``."""
    return _make_loader(data, label, batch_size, L, 3)


# ----------------------------------------------------------------------------
# Loss and training loops
# ----------------------------------------------------------------------------

def listmle_loss(scores, targets):
    """listMLE ranking loss (Cao et al. 2007; Xia et al. 2008).

    L = sum_k log(sum_{j>=k} exp(s_j)) - s_k, via reverse log-cumsum-exp.
    """
    # stable=True keeps the target order deterministic when SIR values tie
    order = targets.flatten().argsort(descending=True, stable=True)
    s = scores[order].flatten()
    rev = torch.flip(torch.logcumsumexp(torch.flip(s, dims=[0]), dim=0), dims=[0])
    return torch.mean(rev - s)


def train_model(loader, model, num_epochs, lr, L, path=None):
    """Standard training loop with MSE regression loss (Adam optimizer)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_list = []
    for epoch in tqdm(range(num_epochs)):
        for data, targets in loader:
            data = data.to(device)
            targets = targets.float().to(device)
            model = model.to(device)
            pred = model(data)
            loss = criterion(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            loss_list.append(float(loss.data))
    if path:
        torch.save(model, path)
    return model, loss_list


def train_model_listmle(loader, model, num_epochs, lr, L, path=None):
    """Training loop with the listMLE loss.

    MSE on the absolute SIR score does not match the Kendall-tau objective.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_list = []
    for epoch in tqdm(range(num_epochs)):
        for data, targets in loader:
            data = data.to(device)
            targets = targets.float().to(device)
            model = model.to(device)
            pred = model(data)
            loss = listmle_loss(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            loss_list.append(float(loss.data))
    if path:
        torch.save(model, path)
    return model, loss_list


# ----------------------------------------------------------------------------
# SIR simulation (ground-truth label generation)
# ----------------------------------------------------------------------------

def SIR(G, infected, beta=0.1, miu=1, N=1000):
    """Run SIR ``N`` times from ``infected``; return the mean final outbreak size.

    Nodes recover with probability ``miu``, so ``miu=1`` means a single step.
    """
    re = 0
    total = N
    while N > 0:
        inf = set(infected)
        R = set()
        while len(inf) != 0:
            newInf = []
            for i in inf:
                for j in G.neighbors(i):
                    k = random.uniform(0, 1)
                    if (k < beta) and (j not in inf) and (j not in R):
                        newInf.append(j)
                k2 = random.uniform(0, 1)
                if k2 < miu:
                    R.add(i)
                else:
                    newInf.append(i)
            inf = set(newInf)
        re += len(R) + len(inf)
        N -= 1
    return re / total if total else 0.0


def SIR_dict(G, beta=0.1, miu=1, real_beta=None, a=1.5):
    """SIR influence of every node (mean over ``N`` runs).

    ``real_beta`` sets beta = a * beta_th with beta_th = <k> / (<k^2> - <k>).
    """
    node_list = list(G.nodes())
    SIR_dic = {}
    if real_beta:
        dc_list = np.array(list(dict(G.degree()).values()))
        beta = a * (float(dc_list.mean()) / (float((dc_list ** 2).mean()) - float(dc_list.mean())))
    print('beta:', beta)
    for node in tqdm(node_list):
        sir = SIR(G, infected=[node], beta=beta, miu=miu)
        SIR_dic[node] = sir
    return SIR_dic


def save_sir_dict(dic, path):
    """Save an SIR dict as a ``{Node, SIR}`` csv file."""
    pd.DataFrame({'Node': list(dic.keys()), 'SIR': list(dic.values())}).to_csv(path, index=False)


def SIR_betas(G, a_list, root_path):
    """Generate SIR labels for every infection-rate ratio in ``a_list`` and
    store them to ``{root_path}{i}.csv``."""
    sir_list = []
    for inx, a in enumerate(a_list):
        sir_dict = SIR_dict(G, real_beta=True, a=a)
        sir_list.append(sir_dict)
        save_sir_dict(sir_dict, root_path + str(inx) + '.csv')
    return sir_list
