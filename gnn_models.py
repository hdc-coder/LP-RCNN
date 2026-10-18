# -*- coding: utf-8 -*-
"""Published GNN baselines used in the paper's comparison.

AGNN  -- Xiong et al., Applied Soft Computing 163 (2024) 111895.
STGCN -- Huang et al., IEEE Trans. Netw. Sci. Eng. 13 (2026) 2350-2370.

Both rank nodes on the *full* graph, trained on BA(1000,4) labels and evaluated
on the nine networks.  AGNN follows the official notebook: a per-network
autoencoder on D^-1/2 (A+I) D^-1/2, and a rank module on the raw A+I trained with
listMLE, scoring each network's own latent.

STGCN deviates from the paper in two places (Eq. (14) input, Eq. (22) skip), so
its numbers are excluded.
"""
import os
import numpy as np
import networkx as nx
import torch
import torch.nn as nn


def build_graph(G, nodes):
    """``(A_norm, D)`` for a full network: the GCN propagation matrix and the
    max-normalised degree vector (the STGCN reconstruction target).
    """
    N = len(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((N, N), dtype=np.float32)
    for u, v in G.edges():
        iu, iv = idx.get(u), idx.get(v)
        if iu is not None and iv is not None:
            A[iu, iv] = 1.0
            A[iv, iu] = 1.0

    deg = np.array([G.degree(n) for n in nodes], dtype=np.float32)
    max_deg = float(deg.max()) if deg.size else 1.0
    D = (deg / max_deg)[:, None]

    A_self = A + np.eye(N, dtype=np.float32)
    dinv = 1.0 / np.sqrt(A_self.sum(1))
    A_norm = (A_self * dinv[:, None]) * dinv[None, :]
    return torch.from_numpy(A_norm).float(), torch.from_numpy(D).float()


class GCNLayer(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        self.lin = nn.Linear(fin, fout)
        self.act = nn.ReLU()

    def forward(self, A, H):
        return self.act(A @ self.lin(H))


def sparse_ai(G, nodes, dense=True):
    """``(A + I)`` in node order and the raw degree column, as the notebook has them.
    """
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    rows, cols = [], []
    for u, v in G.edges():
        i, j = idx.get(u), idx.get(v)
        if i is not None and j is not None:
            rows += [i, j]
            cols += [j, i]
    rows += list(range(n))
    cols += list(range(n))
    deg = np.array([G.degree(v) for v in nodes], dtype=np.float32)[:, None]
    if dense:
        A = np.zeros((n, n), dtype=np.float32)
        np.add.at(A, (rows, cols), 1.0)
        return torch.from_numpy(A), torch.from_numpy(deg)
    import scipy.sparse as sp
    A = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(n, n), dtype=np.float32)
    coo = A.tocoo()
    idx_t = torch.LongTensor(np.vstack([coo.row, coo.col]))
    val_t = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(idx_t, val_t, A.shape).coalesce(), \
        torch.from_numpy(deg)


def sparse_anorm(G, nodes, dense=True):
    """D^-1/2 (A + I) D^-1/2, the propagation of the AGNN autoencoder."""
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    rows, cols = [], []
    for u, v in G.edges():
        i, j = idx.get(u), idx.get(v)
        if i is not None and j is not None:
            rows += [i, j]
            cols += [j, i]
    deg = np.array([G.degree(v) for v in nodes], dtype=np.float64)
    dinv = 1.0 / np.sqrt(deg + 1.0)
    vals = np.array([dinv[i] * dinv[j] for i, j in zip(rows, cols)],
                    dtype=np.float32)
    rows += list(range(n))
    cols += list(range(n))
    diag = np.array([dinv[i] * dinv[i] for i in range(n)], dtype=np.float32)
    if dense:
        A = np.zeros((n, n), dtype=np.float32)
        np.add.at(A, (rows, cols), np.concatenate([vals, diag]))
        return torch.from_numpy(A)
    import scipy.sparse as sp
    A = sp.csr_matrix((np.concatenate([vals, diag]), (rows, cols)),
                      shape=(n, n), dtype=np.float32)
    coo = A.tocoo()
    idx_t = torch.LongTensor(np.vstack([coo.row, coo.col]))
    val_t = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(idx_t, val_t, A.shape).coalesce()


def propagate(A, X):
    """``A @ X`` for dense or sparse COO.

    ``torch.sparse.mm`` is non-deterministic on CUDA, so the dense product is used;
    ``dense=False`` still yields COO.
    """
    return torch.sparse.mm(A, X) if A.is_sparse else A @ X


class _GNNBias(nn.Module):
    """Official CGNN layer: ``relu((A+I) E W + a)``, W and a both xavier."""

    def __init__(self, fin, fout):
        super().__init__()
        self.w = nn.Linear(fin, fout, bias=False)
        self.a = nn.Parameter(torch.zeros(1, fout))
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, A_sp, E):
        return torch.relu(propagate(A_sp, self.w(E)) + self.a)


class AgnAE(nn.Module):
    """AGNN autoencoder: one-hot -> GNN(n,256) -> GNN(256,48) -> fc(48,256) ->
    fc(256,1), reconstructing the degree vector."""

    def __init__(self, n):
        super().__init__()
        self.enc1 = nn.Linear(n, 256, bias=False)
        self.enc2 = nn.Linear(256, 48, bias=False)
        self.dec1 = nn.Linear(48, 256)
        self.dec2 = nn.Linear(256, 1)
        # official GNN layers xavier-initialize their weight matrix
        nn.init.xavier_uniform_(self.enc1.weight)
        nn.init.xavier_uniform_(self.enc2.weight)

    def latent(self, A_sp):
        h = torch.relu(propagate(A_sp, self.enc1.weight.T))  # A_hat W1
        return torch.relu(propagate(A_sp, h @ self.enc2.weight.T))

    def decode(self, Z):
        return self.dec2(self.dec1(Z))                    # no activation

    def recon_loss(self, A_sp, D):
        return torch.sqrt(torch.mean((self.decode(self.latent(A_sp)) - D) ** 2))


class AgnRank(nn.Module):
    """Official AGNN rank module (CGNN): GNN(48,6) -> GNN(6,12) ->
    fc(12,1) with a final ReLU, trained with listMLE."""

    def __init__(self):
        super().__init__()
        self.g1 = _GNNBias(48, 6)
        self.g2 = _GNNBias(6, 12)
        self.fc = nn.Linear(12, 1)

    def forward(self, A_sp, Z):
        x = self.g2(A_sp, self.g1(A_sp, Z))
        return torch.relu(self.fc(x)).flatten()


AGNN_AE_CACHE_VER = 'v2_seeded_init'  # bump whenever AgnAE / its training change
AGNN_AE_CACHE_DIR = os.path.join('cache', f'agnn_ae_{AGNN_AE_CACHE_VER}')


def agnn_ae_cache_path(net_name, seed):
    """Disk cache path for a per-network AGNN autoencoder state dict."""
    return os.path.join(AGNN_AE_CACHE_DIR, f'{net_name}_seed{seed}.pt')


def load_or_train_agnn_ae(net_name, G, nodes, dev, seed, epochs=500,
                          A_norm_sp=None):
    """Per-network AGNN autoencoder, from ``cache/`` when present.

    The seed is set before construction, so cache and fresh training agree.
    """
    path = agnn_ae_cache_path(net_name, seed)
    if os.path.exists(path):
        ae = AgnAE(len(nodes)).to(dev)
        ae.load_state_dict(torch.load(path, map_location=dev,
                                      weights_only=True))
        return ae
    import Utils
    Utils.setup_seed(seed)
    ae = AgnAE(len(nodes)).to(dev)
    if A_norm_sp is None:
        A_norm_sp = sparse_anorm(G, nodes).to(dev)
    _, D = sparse_ai(G, nodes)
    D = D.to(dev)
    train_agnn_ae(ae, A_norm_sp, D, epochs=epochs)
    os.makedirs(AGNN_AE_CACHE_DIR, exist_ok=True)
    torch.save(ae.state_dict(), path)
    return ae


def train_agnn_ae(ae, A_sp, D, epochs=500, verbose=False):
    """Official AE recipe: Adam lr 0.01, wd 5e-4, Frobenius degree
    reconstruction.  The notebook's early-stopping block is dead code, so the
    full budget runs and the final weights are used."""
    import torch.optim as optim
    opt = optim.Adam(ae.parameters(), lr=0.01, weight_decay=5e-4)
    ae.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss = ae.recon_loss(A_sp, D)
        loss.backward()
        opt.step()
    return ae


def train_agnn_rank(rank, A_sp, Z, y, epochs=3000, verbose=False):
    """Official CGNN recipe: Adam lr 0.001, wd 5e-4, listMLE only.  The
    notebook's ``bad_counter`` block never breaks and never restores, so all
    epochs run."""
    import torch.optim as optim
    import Utils
    opt = optim.Adam(rank.parameters(), lr=0.001, weight_decay=5e-4)
    rank.train()
    for ep in range(epochs):
        opt.zero_grad()
        score = rank(A_sp, Z)
        loss = Utils.listmle_loss(score, y)
        loss.backward()
        opt.step()
        if verbose and (ep + 1) % 500 == 0:
            print(f'    rank epoch {ep + 1}/{epochs}  loss = {loss.item():.4f}',
                  flush=True)
    return rank


class SparseTransformerLayer(nn.Module):
    """ST layer (Huang et al. 2026): local attention soft-masked by the
    adjacency, plus global attention to 15 randomly sampled anchors, then a
    residual update and LayerNorm."""
    __constants__ = ['d', 'n_anchors']

    def __init__(self, d, n_anchors=15, dropout=0.1):
        super().__init__()
        self.d = d
        self.n_anchors = n_anchors
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, A, H):
        N = H.size(0)
        Q, K, V = self.wq(H), self.wk(H), self.wv(H)
        # local attention, soft-masked by the adjacency (+ self-loops)
        mask = (A > 0).float() + torch.eye(N, device=H.device)
        scores = Q @ K.T / (self.d ** 0.5)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        out_loc = torch.softmax(scores, dim=-1) @ V
        # global attention through stochastic anchors
        k = min(self.n_anchors, N)
        k = max(2, k)
        anchors = torch.randperm(N, device=H.device)[:k]
        Ka, Va = K[anchors], V[anchors]
        out_glob = torch.softmax(Q @ Ka.T / (self.d ** 0.5), dim=-1) @ Va
        return self.norm(H + self.drop(self.wo(out_loc + out_glob)))


class STGCN(nn.Module):
    """STGCN reproduction (Huang et al. 2026): ST layer + two GCN layers as the
    autoencoder, GNN1/GNN2 + ST layer + linear head as the rank module, listMLE
    only, 15 runs averaged.

    Two deviations keep it out of the paper: a learnable degree-map input instead of
    Eq. (14)'s identity, and no external skip of Eq. (22).
    """
    __constants__ = ['d']

    def __init__(self, d=64, n_anchors=15, dropout=0.1):
        super().__init__()
        self.d = d
        self.embed = nn.Linear(1, d)          # disclosed input adaptation
        self.st_ae = SparseTransformerLayer(d, n_anchors, dropout)
        self.gcn_ae1 = GCNLayer(d, d)
        self.gcn_ae2 = GCNLayer(d, d)
        self.dec1 = nn.Linear(d, d)
        self.dec2 = nn.Linear(d, 1)           # degree reconstruction head
        self.gcn_r1 = GCNLayer(d, d)
        self.gcn_r2 = GCNLayer(d, d)
        self.st_r = SparseTransformerLayer(d, n_anchors, dropout)
        self.head = nn.Linear(d, 1)

    def forward(self, A, D):
        H = self.embed(D)
        Z = self.gcn_ae2(A, self.gcn_ae1(A, self.st_ae(A, H)))
        D_hat = self.dec2(torch.relu(self.dec1(Z)))
        h1 = self.gcn_r1(A, Z)                # Eq. (20): GNN1(Z, Â)
        h2 = self.gcn_r2(A, h1) + h1          # residual: GNN2(h1) + h1
        Hr = self.st_r(A, h2)                 # NOTE: Eq. (22) external skip
                                              # ST(h_gnn2) + h_gnn2 is not
                                              # implemented (parked, see
                                              # class docstring)
        score = self.head(Hr)
        recon_loss = torch.sqrt(torch.mean((D_hat - D) ** 2))
        return score, recon_loss


class Ensemble(nn.Module):
    """Average the scores of several models (the paper's multi-run
    convention for STGCN)."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, A, D):
        scores = [m(A, D)[0] for m in self.members]
        return torch.stack(scores, dim=0).mean(dim=0), None


BUILDERS = {'agnn': AgnRank, 'stgcn': STGCN}


# ---------------------------------------------------------------------------
# Training recipes (from the original publications / official code)
# ---------------------------------------------------------------------------


def train_stgcn(model, A, D, y, ae_epochs=200, rank_epochs=300,
                verbose=False):
    """Two stages: the autoencoder on Frobenius degree reconstruction, then joint
    ranking pretraining of the whole model with listMLE (Adam lr 1e-4).
    """
    import torch
    import torch.optim as optim
    import Utils
    ae_params = (list(model.embed.parameters()) + list(model.st_ae.parameters()) +
                 list(model.gcn_ae1.parameters()) + list(model.gcn_ae2.parameters()) +
                 list(model.dec1.parameters()) + list(model.dec2.parameters()))
    opt_ae = optim.Adam(ae_params, lr=0.01, weight_decay=5e-4)
    sched_ae = optim.lr_scheduler.ReduceLROnPlateau(opt_ae, mode='min',
                                                    factor=0.5, patience=20)
    model.train()
    for ep in range(ae_epochs):
        opt_ae.zero_grad()
        H = model.embed(D)
        Z = model.gcn_ae2(A, model.gcn_ae1(A, model.st_ae(A, H)))
        D_hat = model.dec2(torch.relu(model.dec1(Z)))
        loss = torch.sqrt(torch.mean((D_hat - D) ** 2))
        loss.backward()
        opt_ae.step()
        sched_ae.step(loss)

    # ranking-module pretraining: joint (AE not frozen), listMLE only
    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    for ep in range(rank_epochs):
        opt.zero_grad()
        score, _ = model(A, D)
        loss = Utils.listmle_loss(score.squeeze(1), y)
        loss.backward()
        opt.step()
        if verbose and (ep + 1) % 50 == 0:
            print(f'    rank epoch {ep + 1}/{rank_epochs}  loss = {loss.item():.4f}',
                  flush=True)
    return model
