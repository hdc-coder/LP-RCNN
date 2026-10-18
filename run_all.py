# -*- coding: utf-8 -*-
"""Run the core LP-RCNN pipeline in order.

    1. ``train.py``            train the four CNN models on BA(1000,4)
    2. ``gnn_baselines.py``    train and save AGNN (unless skipped)
    3. ``compare_methods.py``  Experiment 1 -- LP-RCNN vs the baselines
    4. ``ablation_lprcnn.py``  Experiment 2 -- ablation

Robustness, training dynamics and the sweep are opt-in (``--with-robustness`` /
``--with-efficiency`` / ``--with-lr-sweep``); each step is a subprocess logging
to ``logs/``.

    python run_all.py --skip-train --skip-ablation
"""
import os
import sys
import time
import shutil
import argparse
import subprocess
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, 'logs')


def log_path(name):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, f'{stamp}_{name}.log')


def clean():
    """Remove regenerable outputs: figures, trained weights, ablation weights."""
    for d in ('figures',):
        p = os.path.join(ROOT, d)
        if os.path.isdir(p):
            shutil.rmtree(p)
            print(f'  removed {d}/')
    for pat in ('Models/*.pt', 'Models/*.pt.meta.json',
                 'ablation_models/*.pt', 'ablation_models/*.pt.meta.json'):
        for f in glob_pt(pat):
            os.remove(f)
            print(f'  removed {f}')


def glob_pt(pattern):
    import glob
    return glob.glob(os.path.join(ROOT, pattern))


def run_step(name, script, extra_args, env):
    """Run ``python script extra_args`` and stream output to a log file."""
    logf = log_path(name)
    cmd = [sys.executable, '-u', script] + extra_args
    print(f'\n{"=" * 66}\n>>> Step: {name}\n>>> Cmd : {" ".join(cmd)}\n{"=" * 66}')
    t0 = time.time()
    with open(logf, 'w', encoding='utf-8') as fh:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    status = 'OK' if proc.returncode == 0 else f'FAILED (exit {proc.returncode})'
    print(f'>>> {name}: {status}  [{dt / 60:.1f} min]  log: {logf}')
    return proc.returncode == 0


def main():
    ap = argparse.ArgumentParser(description='Run all LP-RCNN experiments')
    ap.add_argument('--clean', action='store_true',
                    help='delete figures/ and trained weights before running')
    ap.add_argument('--skip-train', action='store_true',
                    help='reuse existing Models/*.pt (skip train.py)')
    ap.add_argument('--skip-ablation', action='store_true',
                    help='reuse existing ablation_models/*.pt (skip retraining)')
    ap.add_argument('--gpu', type=int, default=None,
                    help='CUDA device id (sets CUDA_VISIBLE_DEVICES)')
    ap.add_argument('--with-robustness', action='store_true',
                    help='also run robustness.py (multi-seed / multi-network, long)')
    ap.add_argument('--with-efficiency', action='store_true',
                    help='also run time_to_accuracy.py (training dynamics, long)')
    ap.add_argument('--with-lr-sweep', action='store_true',
                    help='also run lr_batch_sweep.py (hyper-parameter grid, very long)')
    ap.add_argument('--skip-gnn', action='store_true',
                    help='do not train/save the AGNN baseline')
    ap.add_argument('--force-gnn', action='store_true',
                    help='retrain AGNN even if Models/agnn.pt already exists')
    args = ap.parse_args()

    env = dict(os.environ)
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'          # avoid OpenMP conflict on Windows
    env['PYTHONUNBUFFERED'] = '1'
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.clean:
        print('Cleaning previous outputs ...')
        clean()

    steps = []
    if not args.skip_train:
        steps.append(('train', 'train.py', []))
    agnn_weights = os.path.join(ROOT, 'Models', 'agnn.pt')
    if (not args.skip_gnn and
            (args.force_gnn or not os.path.exists(agnn_weights))):
        steps.append(('agnn', 'gnn_baselines.py', ['--model', 'agnn']))
    steps.append(('compare', 'compare_methods.py', []))
    if not args.skip_ablation:
        steps.append(('ablation', 'ablation_lprcnn.py', ['--force']))
    else:
        steps.append(('ablation', 'ablation_lprcnn.py', []))
    if args.with_robustness:
        steps.append(('robustness', 'robustness.py', []))
    if args.with_efficiency:
        steps.append(('efficiency', 'time_to_accuracy.py', []))
    if args.with_lr_sweep:
        steps.append(('lr_sweep', 'lr_batch_sweep.py', []))

    print(f'\nPipeline: {[s[0] for s in steps]}')
    ok = True
    for name, script, extra in steps:
        if not run_step(name, os.path.join(ROOT, script), extra, env):
            ok = False
            break

    print('\n' + '=' * 66)
    print('ALL EXPERIMENTS DONE.' if ok else 'PIPELINE ABORTED (see step log).')
    print(f'Figures and tables are in {os.path.join(ROOT, "figures")}.')
    print('=' * 66)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
