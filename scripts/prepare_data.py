#!/usr/bin/env python3
"""
Dataset acquisition + verification for the STAG-STI FSCIL benchmarks.

Places each dataset where the official-split loader (fscil/datasets.py) expects
it, under the repo `data/` directory:

  data/
    cifar-100-python/                         (already present; torchvision)
    MINI-ImageNet/train/<wnid>/<img>.jpg      (miniImageNet, FSCIL layout)
    CUB_200_2011/images/<class>/<img>.jpg     (CUB-200-2011)

Usage:
    python scripts/prepare_data.py --dataset cub200      # auto-download + extract
    python scripts/prepare_data.py --dataset miniimagenet --gdrive-id <ID>
    python scripts/prepare_data.py --dataset miniimagenet   # prints instructions
    python scripts/prepare_data.py --verify all          # check data vs split files

The --verify step confirms that every path/index referenced by the vendored
CEC split files actually resolves in your local data/ tree, so you find missing
files before a multi-hour training run, not during it.
"""

import argparse
import os
import sys
import tarfile
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_DATA = os.path.join(_REPO, "data")

sys.path.insert(0, _REPO)
from fscil.datasets import get_spec, load_official_split, _read_lines, _session_files  # noqa: E402

# Stable public mirror for CUB-200-2011 (Caltech Vision).
CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"

# CEC (Zhang et al., CVPR 2021) Google Drive folder holding miniimagenet.tar +
# CUB_200_2011.tgz, arranged to match the vendored split index files.
CEC_DRIVE_FOLDER = "https://drive.google.com/drive/folders/11LxZCQj2FRCs0JTsf_dafvTHqFn2yGSN"


def _download(url, dest):
    print(f"  downloading {url}\n           -> {dest}")

    def _hook(count, block, total):
        if total > 0:
            pct = min(100, count * block * 100 // total)
            sys.stdout.write(f"\r    {pct:3d}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, _hook)
    print()


def prepare_cub200(force=False):
    target = os.path.join(_DATA, "CUB_200_2011")
    if os.path.isdir(os.path.join(target, "images")) and not force:
        print(f"[cub200] already present at {target}")
        return
    os.makedirs(_DATA, exist_ok=True)
    tgz = os.path.join(_DATA, "CUB_200_2011.tgz")
    if not os.path.exists(tgz):
        try:
            _download(CUB_URL, tgz)
        except Exception as e:  # noqa: BLE001
            print(f"[cub200] automatic download failed: {e}")
            print("  Download CUB_200_2011.tgz manually from:")
            print(f"    {CUB_URL}")
            print(f"  and place it at {tgz}, then re-run.")
            return
    print(f"[cub200] extracting {tgz} ...")
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(_DATA)
    print(f"[cub200] done -> {target}")


def _extract_any(archive, dest):
    """Extract a .tar / .tar.gz / .tgz / .zip into dest."""
    print(f"[extract] {archive} -> {dest}")
    if archive.endswith((".tar", ".tar.gz", ".tgz")):
        mode = "r:gz" if archive.endswith((".gz", ".tgz")) else "r:"
        with tarfile.open(archive, mode) as tf:
            tf.extractall(dest)
    elif archive.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        raise ValueError(f"Unknown archive type: {archive}")


def _normalize_miniimagenet_dirname():
    """The split files expect data/MINI-ImageNet/train/...; if the tar produced
    a differently-cased/spelled top folder, rename it into place."""
    target = os.path.join(_DATA, "MINI-ImageNet")
    if os.path.isdir(os.path.join(target, "train")):
        return
    for cand in ("miniimagenet", "mini_imagenet", "miniImageNet", "MiniImagenet"):
        p = os.path.join(_DATA, cand)
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, "train")):
            print(f"[miniimagenet] renaming {cand} -> MINI-ImageNet")
            os.rename(p, target)
            return


def prepare_miniimagenet(gdrive_id=None, force=False):
    target = os.path.join(_DATA, "MINI-ImageNet")
    if os.path.isdir(os.path.join(target, "train")) and not force:
        print(f"[miniimagenet] already present at {target}")
        return
    try:
        import gdown  # type: ignore
    except ImportError:
        gdown = None

    os.makedirs(_DATA, exist_ok=True)
    if gdown is not None:
        if gdrive_id:
            out = os.path.join(_DATA, "miniimagenet.tar")
            gdown.download(id=gdrive_id, output=out, quiet=False)
            _extract_any(out, _DATA)
        else:
            # pull the whole CEC folder, then extract the miniimagenet tar
            dl = os.path.join(_DATA, "_cec_download")
            print(f"[miniimagenet] downloading CEC drive folder -> {dl}")
            gdown.download_folder(url=CEC_DRIVE_FOLDER, output=dl, quiet=False,
                                  use_cookies=False)
            tar = os.path.join(dl, "miniimagenet.tar")
            if os.path.exists(tar):
                _extract_any(tar, _DATA)
            else:
                print(f"[miniimagenet] miniimagenet.tar not found under {dl}; "
                      "extract it manually into data/.")
        _normalize_miniimagenet_dirname()
        if os.path.isdir(os.path.join(target, "train")):
            print(f"[miniimagenet] ready at {target}. Verify with --verify miniimagenet.")
        else:
            print(f"[miniimagenet] extracted but {target}/train not found -- rename the "
                  "extracted folder to MINI-ImageNet, then --verify miniimagenet.")
        return

    print("[miniimagenet] `pip install gdown` for automatic download, or do it manually:")
    print(f"   1) Open {CEC_DRIVE_FOLDER}")
    print("   2) Download miniimagenet.tar, then:  tar -xvf miniimagenet.tar -C data/")
    print(f"   3) Ensure the layout is {target}/train/<wnid>/<img>.jpg")
    print("   4) python scripts/prepare_data.py --verify miniimagenet")


def verify(dataset_key):
    spec = get_spec(dataset_key)
    print(f"[verify] {spec.pretty_name}")
    if spec.loader == "cifar100":
        import torchvision
        try:
            ds = torchvision.datasets.CIFAR100(root=_DATA, train=True, download=False)
        except Exception as e:  # noqa: BLE001
            print(f"  CIFAR-100 not available under data/: {e}")
            return False
        plan = load_official_split(spec, cifar_targets=ds.targets)
        ok = len(plan["class_order"]) == spec.num_total_classes
        print(f"  class_order={len(plan['class_order'])}/{spec.num_total_classes} "
              f"sessions={[len(s) for s in plan['session_classes']]}  {'OK' if ok else 'MISMATCH'}")
        return ok

    # imagefolder datasets: every referenced relative path must exist under data/
    missing = 0
    total = 0
    for path in _session_files(spec):
        for line in _read_lines(path):
            total += 1
            full = os.path.join(_DATA, line)
            if not os.path.exists(full):
                if missing < 5:
                    print(f"  MISSING: {line}")
                missing += 1
    print(f"  checked {total} referenced files, missing {missing}")
    if missing == 0:
        plan = load_official_split(spec)
        print(f"  class_order={len(plan['class_order'])}/{spec.num_total_classes} "
              f"sessions={[len(s) for s in plan['session_classes']]}  OK")
    return missing == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["cub200", "miniimagenet"],
                    help="which dataset to download/prepare")
    ap.add_argument("--gdrive-id", default=None,
                    help="Google Drive file ID for the miniImageNet archive")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--verify", choices=["cifar100", "miniimagenet", "cub200", "all"],
                    help="verify local data against the vendored split files")
    args = ap.parse_args()

    if args.verify:
        keys = ["cifar100", "miniimagenet", "cub200"] if args.verify == "all" else [args.verify]
        allok = True
        for k in keys:
            try:
                allok &= verify(k)
            except FileNotFoundError as e:
                print(f"  {e}")
                allok = False
        sys.exit(0 if allok else 1)

    if args.dataset == "cub200":
        prepare_cub200(force=args.force)
    elif args.dataset == "miniimagenet":
        prepare_miniimagenet(gdrive_id=args.gdrive_id, force=args.force)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
