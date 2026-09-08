#!/usr/bin/env python3
"""
Prototype rectification (eval-only) to fix the base<->novel imbalance at the
FEATURE level -- unlike TTA (which perturbs inputs and hurt), this recalibrates
the prototypes/queries geometrically.

Variants (all at the final session, best-val ckpts, current novel_logit_bias):
  baseline    : inv_tau * cos(z, H) + bias
  +center     : subtract the base-prototype mean mu before cosine -- removes the
                shared component that inflates base-class similarity (inductive).
  +transductive (BD-CSPN): pseudo-label the unlabelled test queries by nearest
                prototype, then refine each prototype as the normalised mean of
                its memory row + the queries assigned to it, and re-classify.
                NOTE: transductive -- uses the whole test set at once, a
                different (stronger) protocol than inductive per-query eval.

    python scripts/rectify_eval.py --dataset cifar100
"""
import argparse
import os

import torch
import torch.nn.functional as F

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


def hm(a, b):
    return 2 * a * b / (a + b) if (a + b) > 0 else 0.0


def score(pred, true, isbase):
    hit = (pred == true)
    b = hit[isbase].float().mean().item() * 100
    n = hit[~isbase].float().mean().item() * 100
    o = hit.float().mean().item() * 100
    return o, b, n, hm(b, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="BD-CSPN support/query blend")
    args = ap.parse_args()

    spec = Config.apply_dataset(args.dataset)
    set_seed(Config.seed)
    dev = get_device(Config.device)
    plan = load_plan(spec)
    base, sess = get_official_split(spec, plan)
    BASE = set(base)
    all_sessions = [base] + sess

    bb = build_backbone(Config.backbone_type, out_dim=Config.backbone_out_dim).to(dev)
    bb.load_state_dict(torch.load(os.path.join(Config.backbone_ckpt_dir, "backbone_base_best.pt"), map_location=dev)); bb.freeze()
    st = StagStiModel().to(dev)
    st.load_state_dict(torch.load(os.path.join(Config.ckpt_dir, "stag_sti_session0_best.pt"), map_location=dev)); st.eval()
    inv_tau = st.classifier.log_inv_tau.exp().item()

    ds_tr = build_dataset(train=True, allowed_globals=None, transform=None, spec=spec, plan=plan)
    ds_te = build_dataset(train=False, allowed_globals=None, transform=base_transforms(False), spec=spec, plan=plan)
    rng = random.Random(Config.seed + 1000)
    M = Config.num_views

    # final memory H
    H = torch.zeros(0, Config.proj_dim, device=dev); order = []
    for si, cl in enumerate(all_sessions):
        if si == 0:
            P, A = build_new_class_prototypes(bb, st, ds_tr, cl, dev, shot=Config.shot, M=M, rng=rng)
        else:
            refs = official_session_refs(spec, plan, si)
            P, A = build_new_class_prototypes_official(bb, st, ds_tr, refs, cl, dev, M=M)
        H = torch.cat([H, write_new_memory_rows(st, P, A, si, dev)], 0); order.extend(cl)
    nb = len(base)
    novel_col = torch.tensor([c not in BASE for c in order], dtype=torch.float, device=dev)
    bias = Config.novel_logit_bias

    # query features
    pos = {c: i for i, c in enumerate(order)}
    zs, true, isbase = [], [], []
    for c in order:
        idxs = ds_te.indices_for_class(c)
        for _ in idxs:
            true.append(pos[c]); isbase.append(c in BASE)
        imgs = torch.stack([ds_te[i][0] for i in idxs]).to(dev)
        with torch.no_grad():
            zs.append(st.projection(bb(imgs)))
    z = torch.cat(zs, 0)
    true = torch.tensor(true, device=dev); isbase = torch.tensor(isbase, device=dev)

    def logits_plain(zz, HH):
        return inv_tau * (F.normalize(zz, dim=-1) @ F.normalize(HH, dim=-1).T) + novel_col * bias

    print(f"\n[rectify] {spec.pretty_name} (best-val, bias={bias}, inv_tau={inv_tau:.2f})")
    print(f"{'variant':<24}{'overall':>9}{'A_B':>9}{'A_N':>9}{'HM':>9}")

    # baseline
    o, b, n, h = score(logits_plain(z, H).argmax(1), true, isbase); print(f"{'baseline':<24}{o:>9.2f}{b:>9.2f}{n:>9.2f}{h:>9.2f}")

    # +center (subtract base-prototype mean) -- shown to fail (breaks trained cosine geometry)
    mu = H[:nb].mean(0, keepdim=True)
    o, b, n, h = score(logits_plain(z - mu, H - mu).argmax(1), true, isbase); print(f"{'+center (base mu)':<24}{o:>9.2f}{b:>9.2f}{n:>9.2f}{h:>9.2f}")

    # +transductive BD-CSPN on the RAW (calibrated) normalized features, a few iters
    zn, Hn = F.normalize(z, dim=-1), F.normalize(H, dim=-1)
    refined = Hn.clone()
    for _ in range(3):
        pseudo = (inv_tau * (zn @ refined.T) + novel_col * bias).argmax(1)
        newp = refined.clone()
        for k in range(Hn.shape[0]):
            m = (pseudo == k)
            if m.any():
                newp[k] = F.normalize(args.alpha * Hn[k] + (1 - args.alpha) * zn[m].mean(0), dim=-1)
        refined = newp
    o, b, n, h = score((inv_tau * (zn @ refined.T) + novel_col * bias).argmax(1), true, isbase)
    print(f"{'+transductive (raw)':<24}{o:>9.2f}{b:>9.2f}{n:>9.2f}{h:>9.2f}  (transductive)")


if __name__ == "__main__":
    main()
