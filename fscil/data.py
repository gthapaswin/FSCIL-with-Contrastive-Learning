"""
Dataset loading + official FSCIL splits + multi-view "fantasy" augmentation
for STAG-STI, generalized across CIFAR-100 / miniImageNet / CUB-200.

Class identity is a single GLOBAL label in [0, num_total_classes): base
classes first (session 0), then each incremental session's new classes, in
the order fixed by the official CEC split files (see fscil/datasets.py). Base
classes therefore always occupy labels 0..num_base-1, which is exactly what
the Phase-A classifier head expects.

Two dataset backends behind one interface (`IndexedDataset`):
  * cifar100   : wraps torchvision CIFAR-100; split lines are integer indices.
  * imagefolder: miniImageNet / CUB; samples are image files under data/, and
                 a sample's class is its parent folder name.

Both expose: __len__, __getitem__ -> (transformed_tensor, global_label),
get_pil(idx) -> PIL image (for building fantasy views), indices_for_class,
and a `.targets` list of global labels.
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T

from fscil.config import Config
from fscil.datasets import get_spec, load_official_split


# ---------------------------------------------------------------------------
# Official-split plan
# ---------------------------------------------------------------------------
def load_plan(spec=None):
    """Load (and cache) the official-split plan for the active dataset."""
    spec = spec or get_spec(Config.dataset)
    cifar_targets = None
    if spec.loader == "cifar100":
        ds = torchvision.datasets.CIFAR100(root=Config.data_root, train=True, download=True)
        cifar_targets = ds.targets
    return load_official_split(spec, cifar_targets=cifar_targets)


def get_official_split(spec=None, plan=None):
    """Returns (base_classes, incremental_sessions) as lists of GLOBAL labels.
    base_classes = session 0; incremental_sessions = list of per-session lists."""
    spec = spec or get_spec(Config.dataset)
    plan = plan or load_plan(spec)
    sessions = plan["session_classes"]
    return sessions[0], sessions[1:]


# ---------------------------------------------------------------------------
# Transforms (dataset-aware: imagefolder inputs need resizing to image_size)
# ---------------------------------------------------------------------------
def _normalize():
    return T.Normalize(Config.data_mean, Config.data_std)


def base_transforms(train=True, spec=None):
    spec = spec or get_spec(Config.dataset)
    size = spec.image_size
    if spec.loader == "cifar100":
        # native 32x32; no resize needed
        if train:
            return T.Compose([
                T.RandomCrop(size, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                _normalize(),
                T.RandomErasing(p=Config.random_erasing_prob, scale=(0.02, 0.2)),
            ])
        return T.Compose([T.ToTensor(), _normalize()])

    # imagefolder (miniImageNet / CUB): arbitrary-size inputs -> resize
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            _normalize(),
            T.RandomErasing(p=Config.random_erasing_prob, scale=(0.02, 0.2)),
        ])
    return T.Compose([
        T.Resize(int(size * 1.15)),
        T.CenterCrop(size),
        T.ToTensor(),
        _normalize(),
    ])


def make_fantasy_views(pil_img, num_views=None, spec=None):
    """Given a single PIL image, return `num_views` augmented, normalized
    tensor views (M spatial/appearance transforms). Views are resized to the
    dataset's input resolution so the same code works for 32/84/224 px."""
    num_views = num_views or Config.num_views
    spec = spec or get_spec(Config.dataset)
    size = spec.image_size
    norm = _normalize()
    resize = [] if spec.loader == "cifar100" else [T.Resize((size, size))]
    view_transforms = [
        T.Compose([T.RandomResizedCrop(size, scale=(0.8, 1.0)), T.ToTensor(), norm]),
        T.Compose(resize + [T.RandomHorizontalFlip(p=1.0), T.ToTensor(), norm]),
        T.Compose(resize + [T.RandomRotation(15), T.ToTensor(), norm]),
        T.Compose(resize + [T.ColorJitter(0.3, 0.3, 0.3), T.ToTensor(), norm]),
    ]
    views = [view_transforms[m % len(view_transforms)](pil_img) for m in range(num_views)]
    return torch.stack(views, dim=0)  # (M, 3, H, W)


# ---------------------------------------------------------------------------
# Unified dataset
# ---------------------------------------------------------------------------
class IndexedDataset(Dataset):
    """Samples carry GLOBAL labels. Filter to `allowed_globals` (None = all).
    Backend chosen by spec.loader."""

    def __init__(self, spec, plan, train, allowed_globals=None, transform=None, download=True):
        self.spec = spec
        self.transform = transform
        self.loader = spec.loader
        key_to_global = plan["key_to_global"]
        allowed = None if allowed_globals is None else set(int(g) for g in allowed_globals)

        self.data = None          # cifar: numpy array; imagefolder: None
        self.paths = None         # imagefolder: list of file paths
        self.targets = []         # global labels

        if self.loader == "cifar100":
            base = torchvision.datasets.CIFAR100(root=Config.data_root, train=train, download=download)
            data, targets = [], []
            for img, native_label in zip(base.data, base.targets):
                g = key_to_global.get(int(native_label))
                if g is None:
                    continue
                if allowed is not None and g not in allowed:
                    continue
                data.append(img)
                targets.append(g)
            self.data = np.stack(data, axis=0)
            self.targets = targets
        else:
            root = os.path.join(Config.data_root, spec.data_subdir)
            split_dir = "train" if train else "test"
            search_root = os.path.join(root, split_dir)
            if not os.path.isdir(search_root):
                # some datasets (e.g. CUB) store all images under one tree;
                # fall back to the dataset root and rely on folder = class.
                search_root = root
            paths, targets = [], []
            for cls_name, g in key_to_global.items():
                if allowed is not None and g not in allowed:
                    continue
                cls_dir = os.path.join(search_root, str(cls_name))
                if not os.path.isdir(cls_dir):
                    continue
                for fn in sorted(os.listdir(cls_dir)):
                    if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        paths.append(os.path.join(cls_dir, fn))
                        targets.append(g)
            self.paths = paths
            self.targets = targets

        # class -> sample-index list, for episodic / prototype sampling
        self._by_class = {}
        for i, g in enumerate(self.targets):
            self._by_class.setdefault(g, []).append(i)

    def __len__(self):
        return len(self.targets)

    def get_pil(self, idx):
        if self.loader == "cifar100":
            return torchvision.transforms.functional.to_pil_image(self.data[idx])
        return Image.open(self.paths[idx]).convert("RGB")

    def __getitem__(self, idx):
        img = self.get_pil(idx)
        label = self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def indices_for_class(self, global_label):
        return self._by_class.get(int(global_label), [])

    @property
    def classes_present(self):
        return sorted(self._by_class.keys())


def build_dataset(train, allowed_globals=None, transform=None, spec=None, plan=None, download=True):
    spec = spec or get_spec(Config.dataset)
    plan = plan or load_plan(spec)
    return IndexedDataset(spec, plan, train=train, allowed_globals=allowed_globals,
                          transform=transform, download=download)


# ---------------------------------------------------------------------------
# Backward-compatible alias (older code / imports referenced IndexedCIFAR100)
# ---------------------------------------------------------------------------
def IndexedCIFAR100(root, train, class_subset, transform, download=True):
    """Deprecated shim kept for import compatibility. `class_subset` is now
    interpreted as a set of GLOBAL labels (None-equivalent = all classes)."""
    spec = get_spec(Config.dataset)
    plan = load_plan(spec)
    allowed = None if class_subset is None else class_subset
    return IndexedDataset(spec, plan, train=train, allowed_globals=allowed,
                          transform=transform, download=download)


# ---------------------------------------------------------------------------
# Episodic sampler (Phase B) -- generalized over the labels actually present
# ---------------------------------------------------------------------------
import random  # noqa: E402


class EpisodeSampler:
    """Samples `way`-way `shot`-shot + `query`-query episodes from an
    IndexedDataset, over whatever GLOBAL labels are present in it (the base
    pool during Phase B). Support images are expanded into M fantasy views."""

    def __init__(self, dataset: IndexedDataset, way, shot, query, seed=None):
        self.dataset = dataset
        self.way = way
        self.shot = shot
        self.query = query
        self.rng = random.Random(seed)
        self.classes = dataset.classes_present
        self.class_to_indices = {c: dataset.indices_for_class(c) for c in self.classes}

    def sample_episode(self):
        classes = self.rng.sample(self.classes, self.way)
        support_imgs, support_labels = [], []
        query_imgs, query_labels = [], []
        needed = self.shot + self.query
        for local_label, c in enumerate(classes):
            idxs = self.class_to_indices[c]
            if len(idxs) >= needed:
                chosen = self.rng.sample(idxs, needed)
            else:
                chosen = [self.rng.choice(idxs) for _ in range(needed)]
            s_idxs = chosen[:self.shot]
            q_idxs = chosen[self.shot:self.shot + self.query]
            for i in s_idxs:
                views = make_fantasy_views(self.dataset.get_pil(i))  # (M,3,H,W)
                support_imgs.append(views)
                support_labels.append(local_label)
            for i in q_idxs:
                img, _ = self.dataset[i]
                query_imgs.append(img)
                query_labels.append(local_label)
        support_imgs = torch.stack(support_imgs, dim=0)     # (way*shot, M, 3, H, W)
        support_labels = torch.tensor(support_labels)        # (way*shot,)
        query_imgs = torch.stack(query_imgs, dim=0)           # (way*query, 3, H, W)
        query_labels = torch.tensor(query_labels)              # (way*query,)
        return support_imgs, support_labels, query_imgs, query_labels
