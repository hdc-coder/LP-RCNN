# -*- coding: utf-8 -*-
"""Train all neural-network models on the synthetic BA network.

One run per model on ``Networks/training/Train_1000_4.txt`` (BA(1000,4)), saved
as a full model object under ``Models/`` and then evaluated on the nine real
networks by ``compare_methods.py``.

    RCNN     degree                        + MSE       (Yu et al. 2020)
    MRCNN    ND / NCom / k-core            + MSE       (Ou et al. 2022)
    NRCNN    DC / BC / HC + CC contribution + MSE      (Luo & Lu 2023)
    LP-RCNN  ND                            + listMLE

    python train.py --epochs 300 --seed 10
"""
import os
import argparse
import warnings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # avoid OpenMP conflicts on Windows
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
import torch

import Utils
import Embeddings
import Models
import evaluate

L = 28          # neighborhood-network size
LR = 0.005      # LP-RCNN learning rate (grid-selected; baselines use 0.001)
BATCH = 32      # batch size (same as the three CNN baselines)
EPOCHS = 500    # training epochs
SEED = 10

TRAIN_NET_PATH = './Networks/training/Train_1000_4.txt'
TRAIN_SIR_PREFIX = './SIR results/Train_1000_4/BA_1000_4_'
MODEL_DIR = './Models'


def load_training_data():
    """``(G, label)`` for BA(1000,4); the label is ``evaluate.train_label``.
    """
    G = Utils.load_graph(TRAIN_NET_PATH)
    label = evaluate.train_label(TRAIN_SIR_PREFIX)
    missing = set(G.nodes()) ^ set(label.keys())
    if missing:
        raise ValueError(f'training graph / label node mismatch: {len(missing)} nodes')
    return G, label


def build_data(G, label, batch=BATCH):
    """Embeddings and loaders for every model.

    RCNN / M-RCNN / N-RCNN keep their papers' lr 1e-3 and batch 32 (RCNN's paper
    omits the batch, so the family value is used); LP-RCNN uses lr 0.005 / 32.
    """
    _, com, _ = Utils.Louvain(G)
    specs = {
        'rcnn':   (Embeddings.main(G, L),          Utils.Get_DataLoader,   1, 32),
        'mrcnn':  (Embeddings.main1(G, L, com),    Utils.Get_DataLoader1,  3, 32),
        'nrcnn':  (Embeddings.main_n_rcnn(G, L),   Utils.Get_DataLoader1,  3, 32),
        'lprcnn': (Embeddings.main_ndrcnn(G, L),   Utils.Get_DataLoader,   1, batch),
    }
    return {name: (emb, loader_fn(emb, label, b, L), C, b)
            for name, (emb, loader_fn, C, b) in specs.items()}


def make_models():
    """Instantiate every model (fresh weights)."""
    return {
        'rcnn':   Models.CNN(L),
        'mrcnn':  Models.CNN1(L),
        'nrcnn':  Models.N_RCNN(L),
        'lprcnn': Models.DC_RCNN_MS_DIL(L, in_channels=1),   # LP-RCNN final
    }


def main():
    ap = argparse.ArgumentParser(description='Train all LP-RCNN models')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--batch', type=int, default=BATCH)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--only', type=str, default='',
                    help='train only a subset, comma-separated, e.g. lprcnn,rcnn')
    args = ap.parse_args()

    Utils.setup_seed(args.seed)
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f'train.py  device={torch.device("cuda" if torch.cuda.is_available() else "cpu")}  '
          f'epochs={args.epochs}  lr={args.lr}  batch={args.batch}')

    G, label = load_training_data()
    data = build_data(G, label)
    models = make_models()
    if args.only:
        sel = [s.strip() for s in args.only.split(',') if s.strip()]
        data = {k: v for k, v in data.items() if k in sel}
        models = {k: v for k, v in models.items() if k in sel}

    # loss per model: MSE for the published models, listMLE for LP-RCNN.
    # Original-publication hyper-parameters: RCNN / M-RCNN / N-RCNN all use
    # lr 0.001 (and batch 32); LP-RCNN uses this paper's lr 0.005 / batch 32.
    use_listmle = {'lprcnn'}
    model_lr = {'rcnn': 0.001, 'mrcnn': 0.001, 'nrcnn': 0.001}

    for name, model in models.items():
        emb, loader, _, batch = data[name]
        lr = model_lr.get(name, args.lr)
        path = os.path.join(MODEL_DIR, name + '.pt')
        print(f'\n== Training {name}  (loss={"listMLE" if name in use_listmle else "MSE"}, lr={lr}, batch={batch}) ==')
        if name in use_listmle:
            Utils.train_model_listmle(loader, model, args.epochs, lr, L)
        else:
            Utils.train_model(loader, model, args.epochs, lr, L)
        torch.save(model, path)
        metadata = {
            'model': name,
            'loss': 'listMLE' if name in use_listmle else 'MSE',
            'lr': lr,
            'batch_size': batch,
            'epochs': args.epochs,
            'seed': args.seed,
            'train_graph': TRAIN_NET_PATH,
            'train_label_protocol': 'mean SIR over beta/beta_th = 1.0..2.0 (same as every test network)',
            'train_label_files': evaluate.label_files(TRAIN_SIR_PREFIX),
            'train_graph_sha256': Utils.file_sha256(TRAIN_NET_PATH),
            'train_labels_sha256': [Utils.file_sha256(p)
                                    for p in evaluate.label_files(TRAIN_SIR_PREFIX)],
        }
        with open(path + '.meta.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f'   saved -> {path} (+ .meta.json)')

    print('\nAll models trained. Run compare_methods.py / ablation_lprcnn.py / '
          'robustness.py to reproduce the paper results.')


if __name__ == '__main__':
    main()
