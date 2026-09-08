#!/usr/bin/env python3
"""
Eval-only augmentation experiment targeting novel-class accuracy (A_N):
  * support augmentation: build each class's prototype from K stochastic
    augmented copies of every shot (more effective shots -> lower-variance,
    more reliable prototype), then the usual M deterministic fantasy views.
  * test-time augmentation (TTA): average the backbone features of T stochastic
    views of each query before the cosine classifier.

Reports A_B / A_N / HM / overall at the final session for all 4 combinations
(baseline, +support-aug, +TTA, +both), using the best-val checkpoints and the
current novel_logit_bias, so the augmentation effect is isolated. No retraining.

    python scripts/aug_eval.py --dataset cifar100 --supp 5 --tta 5
"""
import argparse
import os
import random

import torch
import torchvision.transforms as T

from fscil.config import Config
from fscil.data import build_dataset, base_transforms, get_official_split, load_plan, make_fantasy_views
from fscil.datasets import official_session_refs
from fscil.backbone import build_backbone
from fscil.pipeline import StagStiModel
from fscil.eval_incremental import write_new_memory_rows, get_device, set_seed


def hm(a, b):
    return 2 * a * b / (a + b) if (a + b) > 0 else 0.0


def _proto_from_pils(backbone, stag, per_class_pils, device, M, chunk):
    """per_class_pils: list over classes of a list of effective-shot PILs.
    Chunked over classes (training scale). Returns P (C,d')."""
    P_parts = []
    for s in range(0, len(per_class_pils), chunk):
        group = per_class_pils[s:s + chunk]
        views = []
        for pils in group:
            views.append(torch.stack([make_fantasy_views(p, M) for p in pils], 0))  # (Keff,M,3,H,W)
        sup = torch.stack(views, 0).to(device)          # (C,Keff,M,3,H,W)
        C, Keff = sup.shape[0], sup.shape[1]
        with torch.no_grad():
            flat = sup.view(C * Keff * M, *sup.shape[-3:])
            feats = backbone(flat).view(C, Keff, M, -1)
            w_cm = feats.mean(dim=1)                       # avg over effective shots
            cid = torch.arange(C, device=device).unsqueeze(1).expand(C, M).reshape(-1)
            vid = torch.arange(M, device=device).unsqueeze(0).expand(C, M).reshape(-1)
            _, h_en, A_soft = stag.encode_prototypes(w_cm.reshape(C * M, -1), cid, vid)
            P, _ = stag.build_prototypes(h_en, A_soft, cid, C)
        P_parts.append(P)
    return torch.cat(P_parts, 0)


def build_memory(backbone, stag, ds, all_sessions, plan, spec, device, shot, M, supp_K, aug):
    chunk = max(1, int(Config.episode_way))
    H = torch.zeros(0, Config.proj_dim, device=device)
    order = []
    for si, cl in enumerate(all_sessions):
        # gather support indices per class
        per_class_idx = []
        if si == 0:
            rng = random.Random(Config.seed + si)
            for c in cl:
                ix = ds.indices_for_class(c)
                per_class_idx.append(rng.sample(ix, shot) if len(ix) >= shot else ix)
        else:
            refs = official_session_refs(spec, plan, si)
            by = {c: [] for c in cl}
            for r in refs:
                i = ds.index_for_ref(r); by[ds.targets[i]].append(i)
            per_class_idx = [by[c] for c in cl]
        # expand each shot into supp_K augmented PILs (K=1 -> raw pil, no aug)
        per_class_pils = []
        for idxs in per_class_idx:
            pils = []
            for i in idxs:
                base_pil = ds.get_pil(i)
                if supp_K <= 1:
                    pils.append(base_pil)
                else:
                    pils.extend(aug(base_pil) for _ in range(supp_K))
            per_class_pils.append(pils)
        P = _proto_from_pils(backbone, stag, per_class_pils, device, M, chunk)
        A = torch.zeros(P.shape[0], P.shape[0], device=device)
        H = torch.cat([H, write_new_memory_rows(stag, P, A, si, device)], 0)
        order.extend(cl)
    return H, order


def query_feats(backbone, stag, ds, order, device, tta, aug, size_tf):
    """Return z (N,d'), true positions, is_base flags. If tta>0, average
    backbone features over tta stochastic views per query."""
    pos = {c: i for i, c in enumerate(order)}
    zs, true, isbase, idxlist = [], [], [], []
    for c in order:
        for i in ds.indices_for_class(c):
            idxlist.append((i, c))
    B = 128
    for s in range(0, len(idxlist), B):
        chunk = idxlist[s:s + B]
        with torch.no_grad():
            if tta and tta > 0:
                acc = None
                for _ in range(tta):
                    imgs = torch.stack([aug(ds.get_pil(i)) for i, _ in chunk]).to(device)
                    f = backbone(imgs)
                    acc = f if acc is None else acc + f
                feat = acc / tta
            else:
                imgs = torch.stack([size_tf(ds.get_pil(i)) for i, _ in chunk]).to(device)
                feat = backbone(imgs)
            zs.append(stag.projection(feat).cpu())
        true.extend(pos[c] for _, c in chunk)
        isbase.extend((c in BASE_SET) for _, c in chunk)
    return torch.cat(zs, 0), torch.tensor(true), torch.tensor(isbase)


def evaluate(z, H, true, isbase, novel_col, bias):
    with torch.no_grad():
        logits = stag_classify(z, H) + novel_col * bias
    pred = logits.argmax(1)
    hit = (pred == true)
    b = hit[isbase].float().mean().item() * 100
    n = hit[~isbase].float().mean().item() * 100
    o = hit.float().mean().item() * 100
    return o, b, n, hm(b, n)


def main():
    global BASE_SET, stag_classify
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--supp", type=int, default=5, help="augmented copies per shot")
    ap.add_argument("--tta", type=int, default=5, help="augmented views per query")
    args = ap.parse_args()

    spec = Config.apply_dataset(args.dataset)
    set_seed(Config.seed)
    device = get_device(Config.device)
    plan = load_plan(spec)
    base, sess = get_official_split(spec, plan)
    BASE_SET = set(base)
    all_sessions = [base] + sess

    backbone = build_backbone(Config.backbone_type, out_dim=Config.backbone_out_dim).to(device)
    backbone.load_state_dict(torch.load(os.path.join(Config.backbone_ckpt_dir, "backbone_base_best.pt"),
                                        map_location=device)); backbone.freeze()
    stag = StagStiModel().to(device)
    stag.load_state_dict(torch.load(os.path.join(Config.ckpt_dir, "stag_sti_session0_best.pt"),
                                    map_location=device)); stag.eval()
    stag_classify = lambda z, H: stag.classify_query(z.to(device), H).cpu()

    ds_tr = build_dataset(train=True, allowed_globals=None, transform=None, spec=spec, plan=plan)
    ds_te = build_dataset(train=False, allowed_globals=None, transform=None, spec=spec, plan=plan)

    size = Config.image_size
    # MILD augmentation -- the frozen backbone is calibrated to near-clean inputs,
    # so gentle crops/flip only (no heavy colour jitter). scale near 1.0.
    aug = T.Compose([T.RandomResizedCrop(size, scale=(0.85, 1.0)), T.RandomHorizontalFlip(),
                     T.ToTensor(), T.Normalize(Config.data_mean, Config.data_std)])
    aug_pil = T.Compose([T.RandomResizedCrop(size, scale=(0.85, 1.0)), T.RandomHorizontalFlip()])
    size_tf = base_transforms(train=False, spec=spec)
    bias = Config.novel_logit_bias

    # two memories
    H_base, order = build_memory(backbone, stag, ds_tr, all_sessions, plan, spec, device,
                                 Config.shot, Config.num_views, supp_K=1, aug=aug_pil)
    H_supp, _ = build_memory(backbone, stag, ds_tr, all_sessions, plan, spec, device,
                             Config.shot, Config.num_views, supp_K=args.supp, aug=aug_pil)
    novel_col = torch.tensor([c not in BASE_SET for c in order], dtype=torch.float)

    # two query-feature sets (no-TTA, TTA)
    z_plain, true, isbase = query_feats(backbone, stag, ds_te, order, device, 0, aug, size_tf)
    z_tta, _, _ = query_feats(backbone, stag, ds_te, order, device, args.tta, aug, size_tf)

    print(f"\n[aug_eval] {spec.pretty_name} (best-val ckpts, bias={bias}, supp K={args.supp}, TTA T={args.tta})")
    print(f"{'config':<22}{'overall':>9}{'A_B':>9}{'A_N':>9}{'HM':>9}")
    for name, z, H in [("baseline", z_plain, H_base), ("+support-aug", z_plain, H_supp),
                       ("+TTA", z_tta, H_base), ("+both", z_tta, H_supp)]:
        o, b, n, h = evaluate(z, H, true, isbase, novel_col, bias)
        print(f"{name:<22}{o:>9.2f}{b:>9.2f}{n:>9.2f}{h:>9.2f}")


if __name__ == "__main__":
    main()
