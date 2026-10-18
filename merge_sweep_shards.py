# -*- coding: utf-8 -*-
"""Merge the sharded screening CSVs into ``lr_batch_sweep.csv``, assert every grid
cell appears once, and redraw the heatmap and Pareto figures.
"""
import os

import pandas as pd

import lr_batch_sweep as sw

SHARDS = ['_shardA', '_shardB', '_shardC']


def main():
    frames = []
    for tag in SHARDS:
        path = os.path.join(sw.OUT_DIR, f'lr_batch_sweep{tag}.csv')
        if not os.path.exists(path):
            print(f'  [warn] missing shard file: {path}')
            continue
        df = pd.read_csv(path)
        print(f'  {tag}: {len(df)} combinations '
              f'(lr {sorted(df.lr.unique())})')
        frames.append(df)
    if not frames:
        raise SystemExit('no shard CSVs found')

    merged = pd.concat(frames, ignore_index=True)
    dup = merged.duplicated(subset=['lr', 'batch']).sum()
    print(f'\nmerged rows: {len(merged)}  duplicates: {dup}')

    expected = {(lr, b) for lr in sw.LRS for b in sw.BATCHES}
    got = set(zip(merged.lr, merged.batch))
    missing = expected - got
    extra = got - expected
    print(f'grid cells expected: {len(expected)}  present: {len(got)}')
    if missing:
        print('  MISSING:', sorted(missing))
    if extra:
        print('  UNEXPECTED:', sorted(extra))

    merged = merged.sort_values(['lr', 'batch']).reset_index(drop=True)
    merged.to_csv(sw.SWEEP_CSV, index=False)
    print(f'\nsaved merged grid: {sw.SWEEP_CSV}')

    ok = merged[merged.status == 'ok']
    print(f'converged: {len(ok)}/{len(merged)}   '
          f'final validation tau: min {ok.final_tau.min():.4f}  '
          f'max {ok.final_tau.max():.4f}')

    sw.analyze(merged, sw.THRESHOLD)
    sw.plot_heatmap(merged)
    sw.plot_pareto(merged)
    print('figures regenerated from the merged grid')


if __name__ == '__main__':
    main()
