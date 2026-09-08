"""
Dataset registry + official FSCIL split parsing for STAG-STI.

This module makes the pipeline dataset-agnostic. Each supported benchmark is
described by a `DatasetSpec` (resolution, normalization, class-split shape,
which backbone stem to use, where its images live, and where its official
CEC split-index files are vendored). It also parses those split files into a
canonical, reproducible class order + per-session sample lists so that our
numbers are directly comparable to FACT / SAVC / CLOSER.

Official splits come from the CEC repo (Zhang et al., CVPR 2021), vendored
under fscil/splits/<dataset>/session_{1..N}.txt:

  * cifar100      : each line is an INTEGER index into the torchvision
                    CIFAR-100 *train* array (fixed pickle order). session_1
                    lists all 30000 base-class train indices (60 classes x
                    500); session_j (j>=2) lists the exact 5-way x 5-shot
                    sample indices for that incremental session.
  * mini_imagenet : each line is a relative image path
                    "MINI-ImageNet/train/<wnid>/<img>.jpg". session_1 =
                    30000 base paths (60 x 500); session_j = 25 few-shot paths.
  * cub200        : each line is a relative image path
                    "CUB_200_2011/images/<cls>/<img>.jpg". session_1 = 3000
                    base paths (100 x 30); session_j = 50 few-shot paths
                    (10-way 5-shot).

The class identity used everywhere downstream is a contiguous GLOBAL label in
[0, num_total_classes): base classes take 0..num_base-1 (in the order they
first appear in session_1), and each incremental session appends its new
classes in split-file order. This global order is fixed and reproducible.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPLIT_ROOT = os.path.join(_HERE, "splits")

# ImageNet statistics (used by miniImageNet + CUB, per FSCIL convention).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# CIFAR-100 statistics.
CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)


@dataclass
class DatasetSpec:
    key: str                       # registry key, e.g. "cifar100"
    pretty_name: str               # human label for logs
    image_size: int                # square input resolution fed to the backbone
    mean: tuple
    std: tuple
    num_base_classes: int          # session-0 class count
    way: int                       # classes added per incremental session
    shot: int                      # support samples per novel class
    num_incremental_sessions: int  # number of sessions AFTER the base session
    backbone: str                  # "cifar_resnet18" | "resnet18" | "resnet18_pretrained"
    split_subdir: str              # folder under fscil/splits/ holding session_*.txt
    loader: str                    # "cifar100" (index-based) | "imagefolder" (path-based)
    data_subdir: str = ""          # for imagefolder datasets: root prefix inside data/
    query_per_class: Optional[int] = None  # eval-time query cap; None = use all test imgs

    @property
    def num_novel_classes(self) -> int:
        return self.way * self.num_incremental_sessions

    @property
    def num_total_classes(self) -> int:
        return self.num_base_classes + self.num_novel_classes

    @property
    def num_sessions(self) -> int:
        """Total sessions including the base session (session 0)."""
        return self.num_incremental_sessions + 1

    @property
    def split_dir(self) -> str:
        return os.path.join(_SPLIT_ROOT, self.split_subdir)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
DATASET_SPECS = {
    "cifar100": DatasetSpec(
        key="cifar100", pretty_name="CIFAR-100",
        image_size=32, mean=CIFAR_MEAN, std=CIFAR_STD,
        num_base_classes=60, way=5, shot=5, num_incremental_sessions=8,
        backbone="cifar_resnet18", split_subdir="cifar100", loader="cifar100",
    ),
    "miniimagenet": DatasetSpec(
        key="miniimagenet", pretty_name="miniImageNet",
        image_size=84, mean=IMAGENET_MEAN, std=IMAGENET_STD,
        num_base_classes=60, way=5, shot=5, num_incremental_sessions=8,
        backbone="resnet18", split_subdir="mini_imagenet", loader="imagefolder",
        data_subdir="MINI-ImageNet",
    ),
    "cub200": DatasetSpec(
        key="cub200", pretty_name="CUB-200-2011",
        image_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD,
        num_base_classes=100, way=10, shot=5, num_incremental_sessions=10,
        backbone="resnet18_pretrained", split_subdir="cub200", loader="imagefolder",
        data_subdir="CUB_200_2011",
    ),
}


def get_spec(dataset_key: str) -> DatasetSpec:
    key = dataset_key.lower().replace("-", "").replace("_", "")
    aliases = {
        "cifar100": "cifar100", "cifar": "cifar100",
        "miniimagenet": "miniimagenet", "mini": "miniimagenet",
        "cub200": "cub200", "cub": "cub200", "cub2002011": "cub200",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown dataset '{dataset_key}'. Valid: {sorted(DATASET_SPECS.keys())}"
        )
    return DATASET_SPECS[aliases[key]]


# ---------------------------------------------------------------------------
# Split parsing
# ---------------------------------------------------------------------------
def _read_lines(path: str):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def _session_files(spec: DatasetSpec):
    """Returns ordered list of session_*.txt paths (session_1..session_N)."""
    files = []
    for i in range(1, spec.num_sessions + 1):
        p = os.path.join(spec.split_dir, f"session_{i}.txt")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing official split file: {p}. Re-download from the CEC repo."
            )
        files.append(p)
    return files


def _class_key_from_line(spec: DatasetSpec, line: str, cifar_targets=None):
    """Map one split-file line to its canonical class key (the identity used
    to build the global class order). For imagefolder datasets that's the
    class folder name; for CIFAR it's the integer label of that sample."""
    if spec.loader == "cifar100":
        idx = int(line)
        return int(cifar_targets[idx])
    # imagefolder: path like ".../<split>/<class>/<img>.jpg" -> class folder
    parts = line.replace("\\", "/").split("/")
    return parts[-2]


def load_official_split(spec: DatasetSpec, cifar_targets=None):
    """
    Parse the vendored CEC split files into a reproducible plan.

    Returns a dict:
      class_order   : list[class_key] of length num_total_classes, in the
                      global order classes are introduced (base first).
      key_to_global : {class_key -> global_label in [0, num_total_classes)}
      session_classes: list of length num_sessions; session_classes[t] is the
                      list of GLOBAL labels introduced at session t (session 0
                      = base classes).
      session_samples: list of length num_sessions; session_samples[t] is the
                      list of raw split-file lines (indices or paths) that make
                      up that session's *support* set. For session 0 this is
                      the full base-train listing; for t>=1 it is the exact
                      few-shot samples defining the episode.

    `cifar_targets` (the torchvision CIFAR-100 train targets array) is required
    only for the cifar100 loader, to turn indices into class labels.
    """
    files = _session_files(spec)
    class_order = []
    seen = set()
    session_classes = []
    session_samples = []

    for t, path in enumerate(files):
        lines = _read_lines(path)
        session_samples.append(lines)
        new_globals = []
        for line in lines:
            ckey = _class_key_from_line(spec, line, cifar_targets)
            if ckey not in seen:
                seen.add(ckey)
                class_order.append(ckey)
        # after registering, compute the global labels new to THIS session
        # (a session introduces classes that first appeared in it)
        session_new_keys = []
        for line in lines:
            ckey = _class_key_from_line(spec, line, cifar_targets)
            if ckey not in session_new_keys:
                session_new_keys.append(ckey)
        session_classes.append(session_new_keys)  # placeholder, remapped below

    key_to_global = {k: i for i, k in enumerate(class_order)}

    # remap session_classes from keys -> global labels, keeping only the keys
    # genuinely NEW at that session (base session keeps all its keys)
    remapped_sessions = []
    already = set()
    for t, keys in enumerate(session_classes):
        new_here = []
        for k in keys:
            if k not in already:
                already.add(k)
                new_here.append(key_to_global[k])
        remapped_sessions.append(sorted(new_here))

    return {
        "class_order": class_order,
        "key_to_global": key_to_global,
        "session_classes": remapped_sessions,
        "session_samples": session_samples,
    }
