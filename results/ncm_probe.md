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

Reading: STAG-STI beats NCM on HM only on miniImageNet; on CIFAR-100 and
CUB-200 the raw-feature NCM is better on every metric. The head currently
loses information that the 512-d backbone features carry (most on CUB).
The reserve backbone transfers worst (A_N 6.8): with Mixup/CutMix off it
reached 99% train accuracy.
