# Synthetic demo record

This page records the first end-to-end synthetic run performed on 2026-08-05.
It is an experiment log, not evidence of real-market performance.

## Environment

| Component | Observed value |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu129 |
| CUDA | 12.9 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| pandas | 2.3.2 |
| NumPy | 2.5.1 |

The environment was prepared without replacing the existing CUDA PyTorch build.
Eight unit tests passed before training.

## Data and split

- Synthetic only; no real prices or paid news content were used.
- 180 trading sessions at five-minute frequency.
- Seven synthetic market series: AAPL, SPY, QQQ, SOXX, TLT, GLD, and USO.
- Final trajectory table: 70,200 rows and 129 columns.
- Last walk-forward fold: 90 train days, 30 validation days, 30 test days.
- Model input used 103 numeric features.

## Model budget

| Setting | Value |
|---|---:|
| Hidden dimension | 48 |
| Transformer layers | 1 |
| Attention heads | 4 |
| Sequence length | 18 |
| Batch size | 128 |
| World Model epochs | 3 |
| Decision Transformer epochs | 4 |

World Model training loss decreased from 0.4943 to 0.3628. Decision Transformer
training loss decreased from 0.7345 to 0.5849; validation loss was approximately
0.589 at the end.

## Outcome and interpretation

Across all tested probability thresholds (0.45 through 0.75), the policy made
zero test-set trades. All 2,340 test bars were classified as `flat`, so exposure,
realized return, and trade count were zero. A hit rate is undefined when there
are no trades.

The only supported conclusion is that this small, short synthetic run did not
produce an actionable policy. Falling training loss does not demonstrate a
profitable strategy, and the all-flat result must not be described as a
successful backtest. Further work requires point-in-time real SIP data, longer
walk-forward evaluation, ablations, cost sensitivity, and enough independent
trading days to estimate uncertainty.

Generated checkpoints and large CSV files are intentionally excluded from Git.
They can be regenerated with `scripts/bootstrap_wsl.sh --with-demo`.
