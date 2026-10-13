# Backbone transfer probe: raw-feature NCM vs STAG-STI (final session)

Plain cosine nearest-class-mean on the frozen 512-d backbone features
(`scripts/ncm_probe.py`), with the same support as STAG-STI: 100 train images
per base class, official few-shot samples for novel classes. Same backbone
checkpoint per row pair. All numbers are final-session accuracy (%).

| Dataset | Backbone | Classifier | A_B | A_N | HM | Final |
|---|---|---|---|---|---|---|
| CIFAR-100 | baseline | NCM | 73.98 | 22.27 | **34.24** | **53.30** |
| CIFAR-100 | baseline | STAG-STI | 68.72 | 20.75 | 31.87 | 49.53 |
| miniImageNet | baseline | NCM | 65.52 | 11.75 | 19.93 | **44.01** |
| miniImageNet | baseline | STAG-STI | 56.12 | 16.45 | **25.44** | 40.25 |
| miniImageNet | bb100 | NCM | 70.72 | 12.35 | 21.03 | **47.37** |
| miniImageNet | bb100 | STAG-STI | 67.72 | 13.00 | **21.81** | 45.83 |
| miniImageNet | reserve100 | NCM | 67.07 | 6.80 | 12.35 | 42.96 |
| miniImageNet | reserve100 | STAG-STI | 62.98 | 7.70 | 13.72 | 40.87 |
| CUB-200 | baseline | NCM | 75.31 | 41.40 | **53.43** | **58.16** |
| CUB-200 | baseline | STAG-STI | 68.99 | 30.85 | 42.64 | 49.71 |

Caveat: the STAG-STI rows above use a 5-shot base draw; the fair base-side
comparison is the full-base run (CIFAR HM 29.77, mini 23.58, CUB 40.01), which
widens the gap on CIFAR/CUB. The probe adds no novel-logit bias, so its A_N is
lower than the evaluator's (see the fusion table below, where both get it).

Reading: STAG-STI beats NCM on HM only on miniImageNet; on CIFAR-100 and
CUB-200 the raw-feature NCM is better on every metric. The head currently
loses information that the 512-d backbone features carry (most on CUB).
The reserve backbone transfers worst (A_N 6.8): with Mixup/CutMix off it
reached 99% train accuracy.

## Residual fusion (`eval_incremental --full_base --fuse ...`)

score = inv_tau * ((1-lam) cos(z, H) + lam cos(f, R)) + novel bias, where
z/H are the STAG-STI query embedding / memory and f/R the raw backbone feature /
class mean. lam=0 is STAG-STI alone, lam=1 plain NCM. Final accuracy / HM (%):

| Dataset | lam=0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|
| CIFAR-100 | 50.44 / 30.78 | 51.57 / 34.35 | 52.37 / 37.91 | **52.86** / 40.49 | 52.58 / **42.26** |
| miniImageNet | 41.40 / 23.80 | 42.62 / 25.38 | 43.20 / 26.87 | **43.50** / 28.05 | 43.40 / **29.19** |
| CUB-200 | 50.10 / 42.20 | 52.52 / 46.66 | 54.47 / 50.27 | **55.25 / 52.25** | 53.52 / 51.41 |

lam=0.75 gives the best final accuracy on all three datasets (STAG-STI adds
information NCM lacks) and +9.7 / +4.3 / +10.1 HM over STAG-STI alone. On HM,
pure NCM is still best on CIFAR and mini. lam was swept on the test set here;
a paper must fix it on held-out base-class validation episodes.

## Backbone comparison on miniImageNet (full-base, novel bias on)

Final acc / A_B / A_N / HM (%) for STAG-STI alone (lam=0), fused (lam=0.75)
and plain NCM (lam=1) on each frozen backbone.

| Backbone | lam=0 (STAG-STI) | lam=0.75 | lam=1 (NCM) |
|---|---|---|---|
| baseline (early-stopped) | 41.40 / 59.07 / 14.90 / 23.80 | 43.50 / 60.32 / 18.28 / 28.05 | 43.40 / 59.43 / 19.35 / 29.19 |
| bb100 (100 ep) | 45.81 / 68.98 / 11.05 / 19.05 | 46.98 / 67.12 / 16.78 / 26.84 | 46.50 / 65.18 / 18.48 / 28.79 |
| reserve_cm100 (100 ep, reserve + CutMix) | 44.15 / 68.13 / 8.17 / 14.60 | 47.55 / 66.65 / 18.90 / 29.45 | **47.69 / 65.30 / 21.28 / 32.09** |

- The reserve + CutMix backbone transfers best: under NCM, +2.8 A_N and +3.3 HM
  over bb100 at the same epochs. This is the best miniImageNet result so far.
- The STAG-STI head degrades as the backbone gets stronger (lam=0 HM 23.8 ->
  19.1 -> 14.6): Phase B, trained on base episodes, collapses novel classes on
  these features. The head, not the backbone, is now the bottleneck.
- Reserve without CutMix (reserve100) overfit to 99% train accuracy and
  transferred worst (NCM HM 12.35).
