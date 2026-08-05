"""
CIFAR-100 loading + base/incremental class split + multi-view "fantasy"
augmentation used to build the multi-view support tensor X_{s,M}.

NOTE ON THE SPLIT
------------------
This reproduces the split logic with a *fixed random seed*: 60 classes are
drawn for the base session, the remaining 40 are shuffled and cut into 8
groups of 5. This is standard-compatible in structure (60 base / 8 sessions
of 5-way-5-shot) but the exact class IDs are this project's own fixed-seed
draw, not TOPIC's official published index list.
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T

from fscil.config import Config


def get_base_novel_split(seed=None):
    seed = seed if seed is not None else Config.seed
    rng = random.Random(seed)
    all_classes = list(range(100))
    rng.shuffle(all_classes)
    base_classes = sorted(all_classes[:Config.num_base_classes])
    novel_classes = all_classes[Config.num_base_classes:]

    sessions = []
    for i in range(Config.num_incremental_sessions):
        start = i * Config.way
        sessions.append(sorted(novel_classes[start:start + Config.way]))
    return base_classes, sessions


class IndexedCIFAR100(Dataset):
    """Wraps torchvision CIFAR100 but only exposes samples whose label is in
    `class_subset`, and remaps labels to a contiguous 0..len(class_subset)-1
    range (needed for classification heads / episodic sampling)."""

    def __init__(self, root, train, class_subset, transform, download=True):
        base = torchvision.datasets.CIFAR100(root=root, train=train, download=download)
        self.transform = transform
        self.label_map = {c: i for i, c in enumerate(sorted(class_subset))}
        self.data = []
        self.targets = []
        for img, label in zip(base.data, base.targets):
            if label in self.label_map:
                self.data.append(img)
                self.targets.append(self.label_map[label])
        self.data = np.stack(self.data, axis=0)
        self.classes_by_new_label = {v: k for k, v in self.label_map.items()}

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]
        img = torchvision.transforms.functional.to_pil_image(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def indices_for_class(self, class_label):
        return [i for i, t in enumerate(self.targets) if t == class_label]


def base_transforms(train=True):
    if train:
        return T.Compose([
            T.RandomCrop(Config.image_size, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(Config.cifar_mean, Config.cifar_std),
            T.RandomErasing(p=Config.random_erasing_prob, scale=(0.02, 0.2)),  # cutout-style regularizer
        ])
    return T.Compose([
        T.ToTensor(),
        T.Normalize(Config.cifar_mean, Config.cifar_std),
    ])


# ---------------------------------------------------------------------------
# Multi-view "fantasy" transforms T_m(.)  (Stage 1 of the spec)
# M deterministic spatial transformations applied to each support image.
# ---------------------------------------------------------------------------
def make_fantasy_views(pil_img_tensor_free, num_views=None):
    """Given a single already-loaded PIL image, returns `num_views` distinct
    deterministic-style augmented tensors (crop/flip/rotate variants), each
    normalized the same way as the base transform."""
    num_views = num_views or Config.num_views
    view_transforms = [
        T.Compose([T.RandomResizedCrop(Config.image_size, scale=(0.8, 1.0)), T.ToTensor(), T.Normalize(Config.cifar_mean, Config.cifar_std)]),
        T.Compose([T.RandomHorizontalFlip(p=1.0), T.ToTensor(), T.Normalize(Config.cifar_mean, Config.cifar_std)]),
        T.Compose([T.RandomRotation(15), T.ToTensor(), T.Normalize(Config.cifar_mean, Config.cifar_std)]),
        T.Compose([T.ColorJitter(0.3, 0.3, 0.3), T.ToTensor(), T.Normalize(Config.cifar_mean, Config.cifar_std)]),
    ]
    # cycle through if num_views > len(view_transforms)
    views = []
    for m in range(num_views):
        tf = view_transforms[m % len(view_transforms)]
        views.append(tf(pil_img_tensor_free))
    return torch.stack(views, dim=0)  # (M, 3, H, W)


class EpisodeSampler:
    """Samples N-way (episode_way) K-shot (episode_shot) + Q-query episodes
    from a given IndexedCIFAR100 dataset, restricted to classes available in
    that dataset (i.e. the base-session pool during Phase B training)."""

    def __init__(self, dataset: IndexedCIFAR100, way, shot, query, seed=None):
        self.dataset = dataset
        self.way = way
        self.shot = shot
        self.query = query
        self.rng = random.Random(seed)
        self.class_to_indices = {}
        for c in range(len(dataset.classes_by_new_label)):
            self.class_to_indices[c] = dataset.indices_for_class(c)

    def sample_episode(self):
        classes = self.rng.sample(list(self.class_to_indices.keys()), self.way)
        support_imgs, support_labels = [], []
        query_imgs, query_labels = [], []
        needed = self.shot + self.query
        for local_label, c in enumerate(classes):
            idxs = self.class_to_indices[c]
            if len(idxs) >= needed:
                chosen = self.rng.sample(idxs, needed)
            else:
                # class pool smaller than shot+query (shouldn't happen on
                # real CIFAR-100, but guards against tiny/custom subsets) --
                # sample with replacement so every episode is still a clean
                # (way*shot) / (way*query) rectangular batch.
                chosen = [self.rng.choice(idxs) for _ in range(needed)]
            s_idxs = chosen[:self.shot]
            q_idxs = chosen[self.shot:self.shot + self.query]
            for i in s_idxs:
                raw_img = self.dataset.data[i]
                pil = torchvision.transforms.functional.to_pil_image(raw_img)
                views = make_fantasy_views(pil)  # (M,3,H,W)
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
