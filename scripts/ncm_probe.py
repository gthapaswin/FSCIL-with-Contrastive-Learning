"""Backbone transfer probe: plain cosine nearest-class-mean on raw frozen
features, no STAG-STI stages. Final session only (all base + novel classes,
official few-shot samples for novel). Separates "the backbone does not
transfer" from "the STAG-STI head loses information".

    python scripts/ncm_probe.py --dataset miniimagenet --tags "" bb100 reserve100
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from fscil.config import Config
from fscil.backbone import build_backbone
from fscil.data import build_dataset, base_transforms, get_official_split, load_plan
from fscil.datasets import official_session_refs
from fscil.train_session0 import get_device


@torch.no_grad()
def extract(backbone, ds, idx, device):
    loader = DataLoader(Subset(ds, idx), batch_size=256, num_workers=Config.num_workers)
    feats, labels = [], []
    for x, y in loader:
        feats.append(F.normalize(backbone(x.to(device)), dim=-1).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="miniimagenet")
    ap.add_argument("--tags", nargs="+", default=[""])
    ap.add_argument("--base_cap", type=int, default=100, help="train images per base class for its mean")
    args = ap.parse_args()

    spec = Config.apply_dataset(args.dataset)
    plan = load_plan(spec)
    base_classes, sessions = get_official_split(spec, plan)
    novel = [c for s in sessions for c in s]
    tf = base_transforms(train=False)
    train_ds = build_dataset(train=True, transform=tf, spec=spec, plan=plan)
    test_ds = build_dataset(train=False, transform=tf, spec=spec, plan=plan)
    support = [i for c in base_classes for i in train_ds.indices_for_class(c)[:args.base_cap]]
    support += [train_ds.index_for_ref(r) for s in range(1, len(sessions) + 1)
                for r in official_session_refs(spec, plan, s)]
    device = get_device(Config.device)
    root = Config.backbone_ckpt_dir

    print(f"{'backbone':<14} {'A_B':>6} {'A_N':>6} {'HM':>6} {'final':>6}")
    for tag in args.tags:
        bb = build_backbone(Config.backbone_type, out_dim=Config.backbone_out_dim).to(device)
        bb.load_state_dict(torch.load(os.path.join(root, tag, "backbone_base.pt"), map_location=device))
        bb.freeze()
        fs, ys = extract(bb, train_ds, support, device)
        ft, yt = extract(bb, test_ds, list(range(len(test_ds))), device)
        classes = base_classes + novel
        protos = F.normalize(torch.stack([fs[ys == c].mean(0) for c in classes]), dim=-1)
        pred = torch.tensor(classes)[(ft @ protos.t()).argmax(1)]
        is_base = torch.isin(yt, torch.tensor(base_classes))
        a_b = (pred[is_base] == yt[is_base]).float().mean().item() * 100
        a_n = (pred[~is_base] == yt[~is_base]).float().mean().item() * 100
        hm = 2 * a_b * a_n / (a_b + a_n) if a_b + a_n else 0.0
        print(f"{tag or 'baseline':<14} {a_b:6.2f} {a_n:6.2f} {hm:6.2f} {(pred == yt).float().mean().item()*100:6.2f}",
              flush=True)


if __name__ == "__main__":
    main()
