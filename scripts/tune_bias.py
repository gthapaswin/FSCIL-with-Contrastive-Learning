#!/usr/bin/env python3
"""
Cheap, no-retraining improvement pass: (1) use the best-val checkpoints, and
(2) sweep the novel-class logit bias to fix the base<->novel imbalance that is
capping A_N / harmonic mean.

Methodology (no test-set tuning): each class's test images are split
deterministically into a VAL half (used only to *select* the bias by harmonic
mean) and a REPORT half (used only to *measure*). We report A_B / A_N / HM /
overall on the REPORT split at three settings: bias=0 (no calibration),
bias=0.5 (the current fixed value), and bias=b* (selected on VAL). The
test-optimal bias is also printed as an unattainable upper bound.

Only the final session (all classes seen) is scored -- that is where the
reported A_B/A_N/HM live. Memory is built once; the bias sweep is then a cheap
add-to-logits, so the whole thing runs in one feature pass per dataset.

    python scripts/tune_bias.py --dataset cifar100
    python scripts/tune_bias.py --dataset cub200
"""
import argparse
import os

import numpy as np
import torch

from fscil.config import Config
from fscil.data import build_dataset, base_transforms, get_official_split, load_plan
from fscil.datasets import official_session_refs
from fscil.backbone import build_backbone
from fscil.pipeline import StagStiModel
from fscil.eval_incremental import (
    build_new_class_prototypes, build_new_class_prototypes_official,
    write_new_memory_rows, get_device, set_seed,
)
import random


def hm(ab, an):
    return 2 * ab * an / (ab + an) if (ab + an) > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--val_frac", type=float, default=0.3)
    args = ap.parse_args()

    spec = Config.apply_dataset(args.dataset)
    set_seed(Config.seed)
    device = get_device(Config.device)
    plan = load_plan(spec)
    base_classes, sessions = get_official_split(spec, plan)
    base_set = set(base_classes)
    all_sessions = [base_classes] + sessions

    # best-val checkpoints
    bpath = os.path.join(Config.backbone_ckpt_dir, "backbone_base_best.pt")
    spath = os.path.join(Config.ckpt_dir, "stag_sti_session0_best.pt")
    print(f"[tune_bias] {spec.pretty_name} | backbone={os.path.basename(bpath)} "
          f"stag={os.path.basename(spath)}")
    backbone = build_backbone(Config.backbone_type, out_dim=Config.backbone_out_dim).to(device)
    backbone.load_state_dict(torch.load(bpath, map_location=device)); backbone.freeze()
    stag = StagStiModel().to(device)
    stag.load_state_dict(torch.load(spath, map_location=device)); stag.eval()

    full_train = build_dataset(train=True, allowed_globals=None, transform=None, spec=spec, plan=plan)
    full_test = build_dataset(train=False, allowed_globals=None,
                              transform=base_transforms(train=False), spec=spec, plan=plan)
    rng = random.Random(Config.seed + 1000)
    M = Config.num_views

    # ---- build final memory H over all classes ----
    H = torch.zeros(0, Config.proj_dim, device=device)
    class_order = []
    for si, cl in enumerate(all_sessions):
        if si == 0:
            P, A = build_new_class_prototypes(backbone, stag, full_train, cl, device,
                                              shot=Config.shot, M=M, rng=rng)
        else:
            refs = official_session_refs(spec, plan, si)
            P, A = build_new_class_prototypes_official(backbone, stag, full_train, refs, cl, device, M=M)
        H = torch.cat([H, write_new_memory_rows(stag, P, A, si, device)], dim=0)
        class_order.extend(cl)
    pos_map = {c: i for i, c in enumerate(class_order)}
    novel_col = torch.tensor([c not in base_set for c in class_order], dtype=torch.float, device=device)

    # ---- one feature pass: logits for every test image ----
    all_logits, all_true, all_isbase, all_isval = [], [], [], []
    for c in class_order:
        idxs = full_test.indices_for_class(c)
        n_val = int(len(idxs) * args.val_frac)
        for j, i in enumerate(idxs):
            all_true.append(pos_map[c]); all_isbase.append(c in base_set)
            all_isval.append(j < n_val)
        imgs = torch.stack([full_test[i][0] for i in idxs]).to(device)
        with torch.no_grad():
            z = stag.projection(backbone(imgs))
            all_logits.append(stag.classify_query(z, H).cpu())
    logits = torch.cat(all_logits, 0)
    true = torch.tensor(all_true)
    isbase = torch.tensor(all_isbase)
    isval = torch.tensor(all_isval)
    ncol = novel_col.cpu()

    def scores(mask, bias):
        lo = logits + ncol * bias
        pred = lo.argmax(1)
        hit = (pred == true)
        b = hit[mask & isbase].float().mean().item() * 100 if (mask & isbase).any() else float("nan")
        n = hit[mask & ~isbase].float().mean().item() * 100 if (mask & ~isbase).any() else float("nan")
        o = hit[mask].float().mean().item() * 100
        return o, b, n, hm(b, n)

    biases = [round(x, 2) for x in np.arange(0.0, 1.51, 0.25)]
    # select bias by HM on VAL
    val_hms = {bz: scores(isval, bz)[3] for bz in biases}
    b_star = max(val_hms, key=val_hms.get)
    # unattainable upper bound: best HM on REPORT itself
    rep_hms = {bz: scores(~isval, bz)[3] for bz in biases}
    b_oracle = max(rep_hms, key=rep_hms.get)

    print(f"\n[tune_bias] REPORT-split results ({spec.pretty_name}, best-val ckpts):")
    print(f"{'setting':<26}{'overall':>9}{'A_B':>9}{'A_N':>9}{'HM':>9}")
    for label, bz in [("bias=0.0 (none)", 0.0), ("bias=0.5 (current)", 0.5),
                      (f"bias={b_star} (val-tuned)", b_star)]:
        o, b, n, h = scores(~isval, bz)
        print(f"{label:<26}{o:>9.2f}{b:>9.2f}{n:>9.2f}{h:>9.2f}")
    o, b, n, h = scores(~isval, b_oracle)
    print(f"{'(oracle bias='+str(b_oracle)+')':<26}{o:>9.2f}{b:>9.2f}{n:>9.2f}{h:>9.2f}  <- upper bound")


if __name__ == "__main__":
    main()
