# STAG-STI — Results & Analysis

_Generated 2026-09-09 05:49 • auto-updated as runs complete._

## In-domain results

| Dataset | A₀ (base) | A_T (final) | PD ↓ | Avg | A_B | A_N | HM |
|---|---|---|---|---|---|---|---|
| CIFAR-100 | 78.22 | 49.53 | 28.69 | 62.08 | 68.72 | 20.75 | 31.87 |
| miniImageNet | 66.13 | 40.25 | 25.88 | 51.18 | 56.12 | 16.45 | 25.44 |
| CUB-200 | 77.79 | 49.71 | 28.09 | 60.88 | 68.99 | 30.85 | 42.64 |

## Cross-domain (miniImageNet → CUB)

- Base (miniImageNet): **66.68%** → final (all seen): **44.27%**, PD **22.41**, A_B 61.38 / A_N 8.41 / HM 14.80

## Comparison vs. published methods (final-session accuracy %)

| Method | CIFAR-100 | miniImageNet | CUB-200 |
|---|---|---|---|
| FACT | TODO | TODO | 56.94 |
| SAVC | TODO | TODO | 62.50 |
| CLOSER | TODO | TODO | 63.58 |
| **STAG-STI (ours)** | **49.53** | **40.25** | **49.71** |

_CUB-200 baselines from the Review-1 deck; CIFAR-100 / miniImageNet baselines are TODO — fill from the source papers._

## CLOSER-style base objective (Δ vs baseline)

| Dataset | Base | Final | A_B | A_N | HM | ΔA_N | ΔHM |
|---|---|---|---|---|---|---|---|
| CIFAR-100 | 71.9 | 46.4 | 61.12 | 24.38 | 34.85 | +3.6 | +3.0 |
| miniImageNet | 63.1 | 37.4 | 53.23 | 13.60 | 21.67 | -2.8 | -3.8 |
| CUB-200 | 74.0 | 48.0 | 64.49 | 31.84 | 42.63 | +1.0 | -0.0 |

_CLOSER helped CIFAR-100 (A_N/HM up) but hurt miniImageNet and was ~neutral on CUB-200 — an inconsistent, dataset-dependent effect at these loss weights (λ_ssl=0.5, λ_close=0.1)._

## Ablation study (CIFAR-100)

| Configuration | A₀ | A_T | PD ↓ | Avg |
|---|---|---|---|---|
| Full model | 78.22 | 49.53 | 28.69 | 62.08 |
| − supcon | 78.53 | 49.28 | 29.25 | 62.00 |
| − topology | 77.75 | 49.69 | 28.06 | 61.65 |
| − agedecay | 78.50 | 49.46 | 29.04 | 62.08 |

## Per-session curves

![combined curves](combined_curves.png)
