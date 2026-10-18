# -*- coding: utf-8 -*-
"""Apply the selection rule of Sec. 4.5.

    1. rank the confirmed configurations by mean validation tau on BA(2000,4);
    2. call one tied if it is within a seed-to-seed std of the best mean;
    3. among the tied, take the lowest mean training time.
"""
import pandas as pd

FILES = ['figures/lr_batch_sweep/lr_batch_confirm_unified.csv']

df = pd.concat([pd.read_csv(f) for f in FILES], ignore_index=True)
df = df.sort_values('val_tau_mean', ascending=False).reset_index(drop=True)

best = df.iloc[0]
tol = best['val_tau_std']
tied = df[df['val_tau_mean'] >= best['val_tau_mean'] - tol]
winner = tied.sort_values('train_sec_mean').iloc[0]

print('confirmed configurations (validation network BA(2000,4)):')
print(df[['lr', 'batch', 'val_tau_mean', 'val_tau_std', 'train_sec_mean']]
      .to_string(index=False))
print(f'\nbest mean      : lr={best.lr:g} batch={int(best.batch)} '
      f'({best.val_tau_mean:.4f} +/- {best.val_tau_std:.4f})')
print(f'tie tolerance  : {tol:.4f} (one std of the best)')
print('tied with best :', [f'{r.lr:g}/{int(r.batch)}' for r in tied.itertuples()])
print(f'\nSELECTED       : lr={winner.lr:g} batch={int(winner.batch)}  '
      f'(val tau {winner.val_tau_mean:.4f} +/- {winner.val_tau_std:.4f}, '
      f'train {winner.train_sec_mean:.0f}s)')
print(f'changes the current paper setting (0.005/32)? '
      f'{"NO" if (winner.lr, int(winner.batch)) == (0.005, 32) else "YES"}')
