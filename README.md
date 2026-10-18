# LP-RCNN

Reference code for *LP-RCNN: A listwise-pooling graph convolutional ranker for
identifying influential nodes in complex networks* (submitted to Neurocomputing).

It contains the training and evaluation scripts behind every table and figure of
the paper, the nine real-world edge lists, the SIR ground-truth labels and the
trained weights.

## 1. Environment

Python 3.11 or 3.12.

| Package | Tested |
|---|---|
| torch | 2.6.0 (CUDA 12.4 build; a CPU build works as well) |
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| networkx | 3.3 |
| scipy | 1.13.1 |
| matplotlib | 3.9.2 |
| seaborn | 0.13.2 |
| python-louvain | 0.16 |
| tqdm | 4.66.5 |

```bash
pip install -r requirements.lock.txt     # exact tested versions
# or
pip install -r requirements.txt          # version ranges
```

Training also runs on CPU; a CUDA GPU only shortens it.

## 2. What each script reproduces

| Script | Output of the paper |
|---|---|
| `run_all.py` | pipeline entry point |
| `train.py` | trains RCNN / M-RCNN / N-RCNN / LP-RCNN into `Models/*.pt` |
| `gnn_baselines.py`, `gnn_models.py` | AGNN baseline |
| `compare_methods.py` | Table 1, Fig. 3, Fig. A.1 |
| `ablation_lprcnn.py` | Table 2 |
| `robustness.py` | Table 3, Table 4 |
| `time_to_accuracy.py` | Fig. 2 |
| `lr_batch_sweep.py`, `select_setting.py`, `merge_sweep_shards.py` | the learning-rate / batch-size selection of Sec. 5.3.1 |
| `generate_sir.py`, `generate_sir_uniform.py` | regeneration of the SIR labels |
| `Models.py`, `Embeddings.py`, `Utils.py`, `evaluate.py` | shared modules |

Fig. 1 (framework) and Table 5 (parameter counts) are not produced by a script.

## 3. Reproduce

Core pipeline:

```bash
python run_all.py
```

`Models/` and `ablation_models/` ship the trained weights, so every table and
figure can also be rebuilt without training:

```bash
python run_all.py --skip-train --skip-ablation
```

The longer studies are opt-in:

```bash
python run_all.py --with-robustness --with-efficiency --with-lr-sweep
```

Other flags: `--skip-train`, `--skip-ablation`, `--skip-gnn`, `--force-gnn`,
`--gpu N`, `--clean`. Run everything from this folder; the scripts resolve their
paths relative to it.

The hyper-parameter sweep has two phases and `select_setting.py` consumes the
output of the second one, so keep this order:

```bash
python lr_batch_sweep.py                                              # grid screening
python lr_batch_sweep.py --confirm "0.001/4,0.002/8"  --tag _conf1    # confirmation
python lr_batch_sweep.py --confirm "0.002/32,0.01/16" --tag _conf2
python select_setting.py                                              # applies the selection rule
```

`merge_sweep_shards.py` is needed only when a screening grid was split across
shards.

## 4. Reproduction requirements

The reported numbers come from one fixed protocol, and the scripts reproduce it
without further configuration:

- training graph BA(1000,4), i.e. `Networks/training/Train_1000_4.txt`;
- neighbourhood size `L = 28`, learning rate `0.005`, batch size `32`,
  500 epochs, seed 10;
- evaluation on the nine real-world networks in `Networks/real/` against the
  SIR labels in `SIR results/`;
- eleven infection ratios, `beta/beta_th = 1.0 ... 2.0`.

`robustness.py` additionally retrains with five seeds and on BA(2000,4) and
BA(3000,4).

## 5. Data

- `Networks/training/` — synthetic BA graphs. `train.py` uses BA(1000,4);
  BA(2000,4) is the held-out validation network of Sec. 5.3.1 and BA(3000,4) is
  used for training-network robustness.
- `Networks/real/` — the nine real-world edge lists.
- `SIR results/<name>/<name>_0.csv … _10.csv` — mean final outbreak size per
  node, one file per infection ratio.
- `Models/` — trained weights, each `.pt` with a `.meta.json` recording its
  training configuration.
- `ablation_models/` — the five ablation checkpoints; rebuild them with
  `ablation_lprcnn.py`.
- `cache/` and `figures/` — created automatically on the first run.

## 6. A note on the SIR labels

These are the label files behind the reported results. Changing a graph file
invalidates its labels.

If you regenerate them, pin `PYTHONHASHSEED`:

```bash
PYTHONHASHSEED=0 python generate_sir.py --network Networks/real/NS.txt \
    --out "SIR results/NS/NS_" --seed 10 --runs 1000
```

## 7. Licence and AI-assisted tools

Released under the terms in [LICENSE](LICENSE).

Parts of the code in this repository were prepared with the assistance of
DeepSeek.
