# STAG-STI — FSCIL with Contrastive Representation Learning

**Spatio-Temporal Graph Integration with Contrastive Pre-Conditioning for Forward-Compatible Few-Shot Class-Incremental Learning.**

A graph-attentive, contrastively pre-conditioned FSCIL system evaluated on three
standard benchmarks (CIFAR-100, miniImageNet, CUB-200) on the **official CEC/FACT
splits**, plus a cross-domain transfer test — with a full ablation and an
evidence-driven improvement study.

> 📊 **Full narrative report (charts + findings):** [`results/report.html`](results/report.html)

---

## What is this?

Few-Shot Class-Incremental Learning (FSCIL) requires a model to keep learning new
categories after deployment — from only a few examples each — without retraining
from scratch and without forgetting. STAG-STI freezes the backbone after the base
session and classifies new classes by nearest-prototype matching, building those
prototypes through a graph-attentive, contrastively pre-conditioned pipeline with
a memory that ages older classes.

### Architecture

```
Fantasy views ×M → Frozen backbone fθ → Projection Wₚ → SupCon head
      → Topology scorer gφ → GATv2 → Prototype pooling → STI memory Hₜ
      → Cosine classifier (queries skip the graph)
```

The three headline components — **contrastive pre-conditioning**, **topology-aware
graph attention**, and the **spatio-temporal (age-decaying) memory** — are each
isolated in the ablation study.

### Benchmarks & protocol

| Dataset | Res | Base / Novel | Sessions | Backbone |
|---|---|---|---|---|
| CIFAR-100 | 32×32 | 60 / 40 | 8 × (5-way 5-shot) | ResNet-18 (CIFAR stem) |
| miniImageNet | 84×84 | 60 / 40 | 8 × (5-way 5-shot) | ResNet-18 |
| CUB-200-2011 | 224×224 | 100 / 100 | 10 × (10-way 5-shot) | ResNet-18 (ImageNet-pretrained) |
| Cross-domain | 84×84 | miniImageNet base → CUB novel | 10 × (10-way 5-shot) | miniImageNet backbone (frozen) |

**Metrics:** Aᵢ (accuracy after session *i* over all classes seen), **PD** = A₀ − A_T
(lower better), **A_B / A_N** (base-only / novel-only accuracy), **HM** (harmonic
mean of A_B and A_N). Everything is frozen at incremental time; no exemplar replay.
Incremental support uses the **exact official few-shot samples**, not random draws.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch, torchvision, numpy, matplotlib

# Datasets (CIFAR-100 auto-downloads via torchvision):
pip install gdown
python scripts/prepare_data.py --dataset cub200         # auto (Caltech mirror)
python scripts/prepare_data.py --dataset miniimagenet   # CEC Google Drive
python scripts/prepare_data.py --verify all             # check vs official splits
```

See [`DATA.md`](DATA.md) for the exact `data/` layout and split details.

---

## Usage

```bash
# Base training (Phase A backbone + Phase B STAG-STI) and incremental eval
python -m fscil.train_session0   --dataset cub200
python -m fscil.eval_incremental --dataset cub200
python -m fscil.eval_incremental --dataset cub200 --transductive   # stronger protocol

# Ablation study (reuses the shared frozen backbone)
python -m fscil.train_session0   --dataset cifar100 --ablate topology --skip_backbone_pretrain
python -m fscil.eval_incremental --dataset cifar100 --ablate topology

# CLOSER-style transfer-oriented base objective
python -m fscil.train_session0   --dataset cifar100 --closer
python -m fscil.eval_incremental --dataset cifar100 --closer

# Cross-domain transfer test (needs the miniImageNet base model)
python -m fscil.eval_crossdomain

# Plots + report data
python -m fscil.plotting --dataset cub200
python scripts/make_report.py            # results/REPORT.md + combined_curves.png
```

Results are **namespaced** so nothing overwrites: baselines in
`checkpoints/<dataset>/`, variants in `ablate-*/`, `closer/`, `closer_lc*/`, and
`transductive/` subdirectories.

---

<!-- AUTO:RESULTS:START -->
## Results

_Auto-generated from `checkpoints/` by `scripts/gen_readme.py` (run on every commit). Single-run, first-pass numbers on official splits._

### In-domain (inductive)

| Dataset | A₀ base | A_T final | PD ↓ | Avg | A_B | A_N | HM |
|---|--:|--:|--:|--:|--:|--:|--:|
| CIFAR-100 | 78.22 | 49.53 | 28.69 | 62.08 | 68.72 | 20.75 | 31.87 |
| miniImageNet | 66.13 | 40.25 | 25.88 | 51.18 | 56.12 | 16.45 | 25.44 |
| CUB-200 | 77.79 | 49.71 | 28.09 | 60.88 | 68.99 | 30.85 | 42.64 |

### Transductive prototype rectification

| Dataset | Final (ind.) | Final (transd.) | A_N (ind.) | A_N (transd.) | HM (ind.) | HM (transd.) |
|---|--:|--:|--:|--:|--:|--:|
| CIFAR-100 | 49.53 | 50.17 | 20.75 | 21.45 | 31.87 | 32.76 |
| miniImageNet | 40.25 | 40.89 | 16.45 | 16.53 | 25.44 | 25.64 |
| CUB-200 | 49.71 | 50.91 | 30.85 | 33.52 | 42.64 | 45.06 |

### Cross-domain (miniImageNet → CUB)

Base (miniImageNet) **66.68%** → final (all seen) **44.27%**, PD **22.41**, A_B 61.38 / A_N 8.41 / HM 14.80.

### Ablation study (CIFAR-100)

| Configuration | A₀ | A_T | A_B | A_N | HM |
|---|--:|--:|--:|--:|--:|
| **Full model** | 78.22 | 49.53 | 68.72 | 20.75 | 31.87 |
| − contrastive | 78.53 | 49.28 | 69.97 | 18.25 | 28.95 |
| − topology | 77.75 | 49.69 | 70.98 | 17.75 | 28.40 |
| − age-decay | 78.50 | 49.46 | 70.15 | 18.43 | 29.18 |

### CLOSER-style base objective (Δ vs baseline)

| Dataset | A_T | A_B | A_N | HM | ΔA_N | ΔHM |
|---|--:|--:|--:|--:|--:|--:|
| CIFAR-100 | 46.42 | 61.12 | 24.38 | 34.85 | +3.6 | +3.0 |
| miniImageNet | 37.38 | 53.23 | 13.60 | 21.67 | -2.8 | -3.8 |
| CUB-200 | 47.98 | 64.49 | 31.84 | 42.63 | +1.0 | -0.0 |

> Full narrative report with charts: **`results/report.html`**.
<!-- AUTO:RESULTS:END -->

---

## Repository layout

```
fscil/
  config.py          # Config + per-dataset switch (apply_dataset) + ablations/CLOSER knobs
  datasets.py        # dataset registry + official CEC split parsing
  data.py            # loaders (cifar / imagefolder / mini_csv), transforms, fantasy views
  backbone.py        # ResNet-18 variants + build_backbone factory
  modules.py         # SupCon, TopologyScorer, GATv2, STI memory, cosine classifier, losses
  pipeline.py        # StagStiModel wiring the stages together
  train_session0.py  # Phase A (backbone) + Phase B (STAG-STI) + CLOSER Phase A
  eval_incremental.py# incremental eval + A_B/A_N/HM + --transductive
  eval_crossdomain.py# miniImageNet → CUB transfer test
  plotting.py        # per-run training / accuracy curves
  splits/            # vendored official CEC index files (cifar100 / mini_imagenet / cub200)
scripts/
  prepare_data.py    # download + verify datasets
  make_report.py     # results/REPORT.md + combined curves
  gen_readme.py      # regenerates the README results block (above)
  run_all.sh / run_closer.sh / run_ablations.sh / rerun_evals.sh  # orchestrators
  tune_bias.py / aug_eval.py / rectify_eval.py                    # improvement experiments
results/
  report.html        # full narrative report
  REPORT.md          # generated results summary
```

---

## Key findings (short version)

- **Novel-class accuracy is the bottleneck** everywhere (A_N 16–31% vs A_B 56–69%).
- **Transductive prototype rectification** is the only reliable improvement (all metrics up; biggest on CUB).
- **Logit-bias** only trades base↔novel; **inference-time augmentation** and **feature centering** don't help.
- **CLOSER-style base training** is dataset-dependent — helped CIFAR, hurt miniImageNet, neutral on CUB.
- Two bugs were caught by verifying results: a CUB base-session collapse (**1% → 78%**) and a cross-domain crash.

Details, tables and charts: [`results/report.html`](results/report.html).

---

## Auto-updating README

The **Results** section above regenerates from `checkpoints/` on every commit, via a
git `pre-commit` hook. Enable it once per clone:

```bash
git config core.hooksPath .githooks
```

After that, any commit (following a new run or a code change) refreshes the results
tables automatically. Run it manually anytime with `python scripts/gen_readme.py`.

---

## Notes

- Numbers are **single-run, first-pass** on Apple-Silicon (mps) with official splits and deterministic fantasy views — internally consistent and reproducible, but not tuned to compete with published SOTA and not seed-averaged.
- Published CIFAR-100 / miniImageNet baselines are intentionally left for you to fill from the source papers.
- The cross-domain protocol is this project's own definition, not a citable standard.
