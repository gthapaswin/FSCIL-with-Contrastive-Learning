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

## Held-out Phase B on the reserve + CutMix backbone (`--heldout`, mini)

Phase B trained on 45 base classes with the residual fused into its loss
(lam_train 0.5); checkpoint and lam selected on 15 held-out base classes, not
the test set. Test (full-base, novel bias), final acc / A_B / A_N / HM (%):

| lam | 0 | 0.25 | 0.5 | **0.75 (val-selected)** | 1 (NCM) |
|---|---|---|---|---|---|
| held-out Phase B | 0.96 / 0.00 / 2.40 / 0.00 | 43.23 / 53.25 / **28.20 / 36.87** | 46.77 / 61.38 / 24.85 / 35.38 | **47.34 / 63.70 / 22.80 / 33.58** | 47.60 / 64.88 / 21.67 / 32.49 |

- At the validation-chosen lam=0.75 the head beats plain NCM on HM (+1.1) and
  A_N (+1.1) at -0.3 final accuracy. Lower lam trades final accuracy for HM
  (lam=0.25: HM 36.87, A_N 28.20, the best novel accuracy on mini so far).
- The held-out validation is nearly flat across lam=0.5-1 (HM 74.3-74.5),
  because the backbone saw the held-out classes in Phase A: it cannot rank
  lam the way the true novel test does.
- The best checkpoint is epoch 2 (train query accuracy is already 98.8% at
  epoch 1): Phase B fits base episodes almost immediately.
- With residual training, the STAG-STI branch alone (lam=0) is not a usable
  classifier; it only works as a residual.

## Recipe on CIFAR-100 (`scripts/run_recipe.sh cifar100 100`)

Reserve + CutMix backbone (100 ep), held-out Phase B (best epoch 1), full-base
fused eval. Headline lam=0.5 is fixed in advance (= the training lambda); the
held-out validation also selected 0.5. Final acc / A_B / A_N / HM (%):

| lam | 0.25 | **0.5 (headline)** | 0.75 | 1 (NCM, same backbone) |
|---|---|---|---|---|
| CIFAR-100 | 52.41 / 60.42 / 40.40 / 48.42 | **56.30 / 70.37 / 35.20 / 46.93** | 57.01 / 73.45 / 32.35 / 44.92 | 57.26 / 74.72 / 31.08 / 43.89 |

- +3.0 HM and +4.1 A_N over NCM on the same backbone, at -1.0 final accuracy
  (miniImageNet: +2.9 HM at -0.8 final).
- vs the Review II result (final 49.53, HM 31.87): +6.8 final, +15.1 HM.
- The standard (non-held-out) Phase B on this backbone collapses (HM 15.01).

## Recipe on CUB-200 (`scripts/run_recipe.sh cub200 30`)

Reserve + CutMix backbone (30 ep, ImageNet-pretrained ResNet-18), held-out
Phase B (best epoch 1; held-out HM falls from 80.6 to ~77 afterwards), full-base
fused eval. Held-out validation selected lam=0.75. Final acc / A_B / A_N / HM (%):

| lam | 0.25 | **0.5 (headline)** | 0.75 (val-selected) | 1 (NCM, same backbone) |
|---|---|---|---|---|
| CUB-200 | 32.02 / 19.20 / 44.54 / 26.84 | **49.28 / 54.40 / 44.27 / 48.81** | 53.69 / 64.18 / 43.45 / 51.82 | 55.51 / 68.72 / 42.59 / 52.59 |

- **Negative on CUB:** the residual head is below NCM on the same backbone
  (-3.8 HM at lam=0.5, -0.8 at the val-selected 0.75). A_N gains only +1.7
  while A_B drops 14 points: on fine-grained classes with a pretrained
  backbone the raw-feature means are already strong and the head mostly
  adds base-class confusion.
- The held-out validation is flat over lam=0.5-1 (80.6/80.8/80.3) and cannot
  detect this.
- Standard Phase B on this backbone: final 50.24, HM 42.95.
- lam=0 is degenerate (final 0.43): as on mini, the residual-trained branch is
  not a standalone classifier.
