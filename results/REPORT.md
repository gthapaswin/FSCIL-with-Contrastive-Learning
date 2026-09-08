# STAG-STI — Results & Analysis

_Generated 2026-09-08 18:09 • auto-updated as runs complete._

## In-domain results

| Dataset | A₀ (base) | A_T (final) | PD ↓ | Avg | A_B | A_N | HM |
|---|---|---|---|---|---|---|---|
| CIFAR-100 | 78.23 | 49.43 | 28.80 | 61.98 | 68.45 | 20.90 | 32.02 |
| miniImageNet | 66.57 | 40.34 | 26.23 | 51.32 | 56.20 | 16.55 | 25.57 |
| CUB-200 | 1.05 | 1.73 | -0.68 | 5.39 | 0.00 | 3.41 | 0.00 |

## Cross-domain (miniImageNet → CUB)

- _pending_

## Comparison vs. published methods (final-session accuracy %)

| Method | CIFAR-100 | miniImageNet | CUB-200 |
|---|---|---|---|
| FACT | TODO | TODO | 56.94 |
| SAVC | TODO | TODO | 62.50 |
| CLOSER | TODO | TODO | 63.58 |
| **STAG-STI (ours)** | **49.43** | **40.34** | **1.73** |

_CUB-200 baselines from the Review-1 deck; CIFAR-100 / miniImageNet baselines are TODO — fill from the source papers._

## Ablation study (CIFAR-100)

| Configuration | A₀ | A_T | PD ↓ | Avg |
|---|---|---|---|---|
| Full model | 78.23 | 49.43 | 28.80 | 61.98 |
| − supcon | 78.40 | 48.76 | 29.64 | 61.71 |
| − topology | _pending_ | | | |
| − agedecay | _pending_ | | | |

## Per-session curves

![combined curves](combined_curves.png)
