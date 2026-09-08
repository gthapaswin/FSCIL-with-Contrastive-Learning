# Datasets & official FSCIL splits

STAG-STI is evaluated on three in-domain benchmarks plus a cross-domain
transfer test. All use the **official CEC/FACT class splits** (vendored under
`fscil/splits/`) so results are directly comparable to FACT / SAVC / CLOSER.

| Dataset | Res | Base / Novel | Sessions | Backbone |
|---|---|---|---|---|
| CIFAR-100 | 32×32 | 60 / 40 | 8 × (5-way 5-shot) | ResNet-18 (CIFAR stem) |
| miniImageNet | 84×84 | 60 / 40 | 8 × (5-way 5-shot) | ResNet-18 |
| CUB-200-2011 | 224×224 | 100 / 100 | 10 × (10-way 5-shot) | ResNet-18 (ImageNet-pretrained) |
| Cross-domain | — | miniImageNet base → CUB novel | — | miniImageNet backbone |

## Expected `data/` layout

```
data/
  cifar-100-python/                       # torchvision auto-download (present)
  MINI-ImageNet/train/<wnid>/<img>.jpg    # miniImageNet, FSCIL layout
  CUB_200_2011/images/<class>/<img>.jpg   # CUB-200-2011
```

`data/` is git-ignored — datasets are never committed. Only the small split
index files (`fscil/splits/`) are tracked.

## Getting the data

```bash
# CUB-200-2011 — automatic (stable Caltech mirror, ~1.1 GB)
python scripts/prepare_data.py --dataset cub200

# miniImageNet — distributed via Google Drive by the CEC/CLOSER authors.
# Grab the file ID and:
python scripts/prepare_data.py --dataset miniimagenet --gdrive-id <ID>
# (or arrange images manually into data/MINI-ImageNet/train/<wnid>/<img>.jpg)

# Verify local data matches the official split files BEFORE training:
python scripts/prepare_data.py --verify all
```

## Official splits

Vendored from the CEC repo (`icoz69/CEC-CVPR2021`, `data/index_list/`):

- `fscil/splits/cifar100/session_{1..9}.txt` — integer indices into the
  torchvision CIFAR-100 **train** array.
- `fscil/splits/mini_imagenet/session_{1..9}.txt` — relative image paths.
- `fscil/splits/cub200/session_{1..11}.txt` — relative image paths.

`session_1.txt` is the full base-session training set; `session_j` (j≥2) lists
the exact k-shot support samples for that incremental session. Parsing lives in
`fscil/datasets.py` (`load_official_split`), which builds a fixed, reproducible
global class order (base classes first, then each session's new classes).
