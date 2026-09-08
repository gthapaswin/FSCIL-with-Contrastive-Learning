"""
Full FSCIL incremental-session evaluation for STAG-STI on CIFAR-100.

Loads the frozen backbone + trained Session-0 STAG-STI weights, then replays
the 8 incremental sessions (5-way 5-shot each):

  Session 0: persistent memory H_0 is built from the 60 base classes
             (5-shot prototypes each, matching the spec's own worked
             example table: C_t=60, S=5, M=4), then evaluated on the full
             CIFAR-100 test set restricted to those 60 classes.
  Session t (1..8): 5 new classes' 5-shot support is turned into
             prototypes via the SAME frozen graph pipeline (Stages 3-6),
             merged into persistent memory via STI (Stage 7), and the
             model is evaluated on the test set across ALL classes seen
             so far (base + every novel class introduced up to t).

No gradients anywhere in this script -- everything is frozen per the
"Optimizer: None" / zero-gradient-updates rule for incremental sessions.

# ASSUMPTION: the topology/GATv2 graph for each session's new-class
# prototypes is built ONLY among that session's own M-view nodes, not
# jointly attended with existing memory rows. This is a deliberate
# simplification -- jointly re-attending the full growing class pool every
# session is expensive and, more importantly, is mathematically moot for
# THIS script's purpose: a class's memory row is written exactly once (at
# its introduction session) and never revisited, so by the time
# STIMemory.forward() runs for a batch of brand-new classes, new_class_mask
# forces H_t = Phi(P_t) regardless of R_gate/Gamma -- the retention gate
# literally cannot affect a class's very first (and only) memory write.
# R_gate only has teeth if a class's row is recomputed in a LATER session,
# which standard FSCIL protocol does not do. If your setup expects
# cross-session joint graph attention at eval time too, tell me and I'll
# extend this.

Run:
    python -m fscil.eval_incremental
    python -m fscil.eval_incremental --backbone_ckpt checkpoints/backbone_base_best.pt \
                                      --stag_ckpt checkpoints/stag_sti_session0_best.pt
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fscil.config import Config
from fscil.data import build_dataset, base_transforms, get_official_split, make_fantasy_views, load_plan
from fscil.datasets import official_session_refs
from fscil.backbone import build_backbone
from fscil.pipeline import StagStiModel
from fscil.train_session0 import get_device, set_seed

import torchvision


def _prototypes_one_group(backbone, stag_model, full_train_ds, per_class_indices, device, M):
    """Build prototypes for a small group of classes in a single graph."""
    per_class_views = []
    for idxs in per_class_indices:
        shot_views = [make_fantasy_views(full_train_ds.get_pil(i), M) for i in idxs]  # each (M,3,H,W)
        per_class_views.append(torch.stack(shot_views, dim=0))                          # (shot,M,3,H,W)
    support_imgs = torch.stack(per_class_views, dim=0).to(device)  # (way, shot, M, 3, H, W)
    way, shot = support_imgs.shape[0], support_imgs.shape[1]

    with torch.no_grad():
        flat = support_imgs.view(way * shot * M, *support_imgs.shape[-3:])
        feats = backbone(flat).view(way, shot, M, -1)      # (way, shot, M, 512)
        w_cm = feats.mean(dim=1)                              # (way, M, 512) -- Stage 3 shot-averaging

    class_ids = torch.arange(way, device=device).unsqueeze(1).expand(way, M).reshape(-1)
    view_ids = torch.arange(M, device=device).unsqueeze(0).expand(way, M).reshape(-1)
    node_feats = w_cm.reshape(way * M, -1)

    with torch.no_grad():
        _, h_enriched, A_soft = stag_model.encode_prototypes(node_feats, class_ids, view_ids)
        P_new, A_tilde_new = stag_model.build_prototypes(h_enriched, A_soft, class_ids, way)
    return P_new, A_tilde_new


def _prototypes_from_indices(backbone, stag_model, full_train_ds, per_class_indices, device, M):
    """per_class_indices: list (one entry per class, in class order) of the
    sample indices to use as that class's support shots. Runs Stages 1-6.

    The class graph is built in chunks of at most Config.episode_way classes --
    the scale the model was trained on. A single monolithic graph over a large
    base session (e.g. 100 classes -> 400 nodes) is both out-of-distribution
    for the GATv2 (trained on ~15-way episodes) and numerically unreliable on
    some backends; chunking fixes both. This does not change the memory write:
    new-class rows are H = Phi(P) regardless of the class adjacency (see
    write_new_memory_rows / STIMemory.new_class_mask), so A_tilde here is a
    placeholder for brand-new classes."""
    n = len(per_class_indices)
    chunk = max(1, int(getattr(Config, "episode_way", 15)))
    if n <= chunk:
        return _prototypes_one_group(backbone, stag_model, full_train_ds, per_class_indices, device, M)
    P_parts = []
    for start in range(0, n, chunk):
        P_g, _ = _prototypes_one_group(backbone, stag_model, full_train_ds,
                                       per_class_indices[start:start + chunk], device, M)
        P_parts.append(P_g)
    P_new = torch.cat(P_parts, dim=0)                       # (n, d')
    A_tilde_new = torch.zeros(n, n, device=device)          # unused for new-class writes
    return P_new, A_tilde_new


def build_new_class_prototypes(backbone, stag_model, full_train_ds, class_list, device, shot, M, rng):
    """Random k-shot support (used for the base session). class_list: GLOBAL
    labels introduced this session. Returns P_new, A_tilde_new."""
    per_class_indices = []
    for c in class_list:
        idxs = full_train_ds.indices_for_class(c)
        chosen = rng.sample(idxs, shot) if len(idxs) >= shot else [rng.choice(idxs) for _ in range(shot)]
        per_class_indices.append(chosen)
    return _prototypes_from_indices(backbone, stag_model, full_train_ds, per_class_indices, device, M)


def build_new_class_prototypes_official(backbone, stag_model, full_train_ds, refs, class_list, device, M):
    """Reproducible support using the EXACT few-shot samples from the official
    split file (fscil.datasets.official_session_refs). `refs` are grouped by
    class and ordered to match `class_list`."""
    by_class = {c: [] for c in class_list}
    for ref in refs:
        idx = full_train_ds.index_for_ref(ref)
        by_class[full_train_ds.targets[idx]].append(idx)
    per_class_indices = [by_class[c] for c in class_list]
    return _prototypes_from_indices(backbone, stag_model, full_train_ds, per_class_indices, device, M)


def write_new_memory_rows(stag_model, P_new, A_tilde_new, session_idx, device):
    way = P_new.shape[0]
    dim = P_new.shape[1]
    H_prev_dummy = torch.zeros(way, dim, device=device)
    class_ages = torch.full((way,), float(session_idx), device=device)
    new_class_mask = torch.ones(way, dtype=torch.bool, device=device)
    with torch.no_grad():
        H_new_rows, _, _ = stag_model.update_memory(H_prev_dummy, P_new, A_tilde_new, class_ages,
                                                        session=session_idx, new_class_mask=new_class_mask)
    return H_new_rows


@torch.no_grad()
def evaluate_cumulative(backbone, stag_model, full_test_ds, class_order, H, device,
                         novel_mask=None, base_set=None, batch_size=256):
    """Accuracy over the test set restricted to every class in class_order,
    classified against the current persistent memory H (rows aligned to
    class_order).

    novel_mask: optional (len(class_order),) bool tensor, True for classes
    NOT introduced in the base session. When given, Config.novel_logit_bias
    is added to those classes' logits before argmax -- a calibration
    correction for the systematic base-class favoritism that comes from
    base prototypes being far better estimated (many images) than 5-shot
    novel ones, even after cosine normalization removes any raw magnitude
    effect. Set Config.novel_logit_bias = 0.0 to disable.
    """
    pos_map = {c: i for i, c in enumerate(class_order)}
    indices = []
    for c in class_order:
        indices.extend(full_test_ds.indices_for_class(c))

    correct, total = 0, 0
    base_correct, base_total = 0, 0
    novel_correct, novel_total = 0, 0
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        imgs, true_pos, is_base = [], [], []
        for i in batch_idx:
            img, raw_label = full_test_ds[i]
            imgs.append(img)
            true_pos.append(pos_map[raw_label])
            is_base.append(base_set is None or raw_label in base_set)
        imgs = torch.stack(imgs, dim=0).to(device)
        true_pos = torch.tensor(true_pos, device=device)

        feats = backbone(imgs)
        z_q = stag_model.projection(feats)
        logits = stag_model.classify_query(z_q, H)
        if novel_mask is not None and Config.novel_logit_bias != 0.0:
            logits = logits + novel_mask.to(device).float() * Config.novel_logit_bias
        preds = logits.argmax(dim=1)
        hit = (preds == true_pos)
        correct += hit.sum().item()
        total += len(batch_idx)
        for j, b in enumerate(is_base):
            if b:
                base_total += 1
                base_correct += int(hit[j].item())
            else:
                novel_total += 1
                novel_correct += int(hit[j].item())
    return {
        "acc": correct / total,
        "acc_base": (base_correct / base_total) if base_total else float("nan"),
        "acc_novel": (novel_correct / novel_total) if novel_total else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar100",
                         help="cifar100 | miniimagenet | cub200")
    parser.add_argument("--ablate", nargs="+", default=None,
                         choices=["none", "supcon", "topology", "agedecay"],
                         help="Must match the ablations used at training time "
                              "(loads from and writes to the ablate-... namespace).")
    parser.add_argument("--shot", type=int, default=None)
    parser.add_argument("--random_support", action="store_true",
                         help="Sample incremental support randomly instead of using the exact "
                              "official few-shot samples (default: use official samples).")
    parser.add_argument("--backbone_ckpt", type=str, default=None,
                         help="Path to backbone checkpoint. Defaults to checkpoints/backbone_base.pt "
                              "(final-epoch). Pass checkpoints/backbone_base_best.pt to use the best-val "
                              "checkpoint instead.")
    parser.add_argument("--stag_ckpt", type=str, default=None,
                         help="Path to STAG-STI checkpoint. Defaults to checkpoints/stag_sti_session0.pt "
                              "(final-epoch). Pass checkpoints/stag_sti_session0_best.pt to use the "
                              "best-val checkpoint instead.")
    args = parser.parse_args()

    spec = Config.apply_dataset(args.dataset)
    ablations = Config.apply_ablation(args.ablate)
    shot = args.shot if args.shot is not None else Config.shot
    set_seed(Config.seed)
    device = get_device(Config.device)
    print(f"Using device: {device}")
    print(f"Dataset: {spec.pretty_name} | backbone: {Config.backbone_type}"
          + (f" | ABLATION: {ablations}" if ablations else ""))

    plan = load_plan(spec)
    base_classes, incremental_sessions = get_official_split(spec, plan)
    base_set = set(base_classes)
    all_sessions = [base_classes] + incremental_sessions

    backbone_ckpt = args.backbone_ckpt or os.path.join(Config.backbone_ckpt_dir, "backbone_base.pt")
    stag_ckpt = args.stag_ckpt or os.path.join(Config.ckpt_dir, "stag_sti_session0.pt")
    if not (os.path.exists(backbone_ckpt) and os.path.exists(stag_ckpt)):
        raise FileNotFoundError(
            f"Missing checkpoints. Expected:\n  {backbone_ckpt}\n  {stag_ckpt}\n"
            "Run `python -m fscil.train_session0` first."
        )

    backbone = build_backbone(Config.backbone_type, out_dim=Config.backbone_out_dim).to(device)
    backbone.load_state_dict(torch.load(backbone_ckpt, map_location=device))
    backbone.freeze()

    stag_model = StagStiModel().to(device)
    stag_model.load_state_dict(torch.load(stag_ckpt, map_location=device))
    stag_model.eval()

    full_train_ds = build_dataset(train=True, allowed_globals=None, transform=None,
                                  spec=spec, plan=plan)
    full_test_ds = build_dataset(train=False, allowed_globals=None,
                                 transform=base_transforms(train=False), spec=spec, plan=plan)

    rng = random.Random(Config.seed + 1000)

    H = torch.zeros(0, Config.proj_dim, device=device)
    class_order = []
    results = []

    print(f"\n=== Incremental FSCIL evaluation: {len(all_sessions)} sessions "
          f"(session 0 = {len(base_classes)} base classes, sessions 1-{spec.num_incremental_sessions} "
          f"= {spec.way}-way {shot}-shot) ===")

    use_official = not args.random_support
    for session_idx, class_list in enumerate(all_sessions):
        t0 = time.time()
        # Incremental sessions (>=1) use the exact official few-shot samples for
        # reproducibility; the base session builds its prototypes from a random
        # k-shot draw of the abundant base training data.
        if use_official and session_idx >= 1:
            refs = official_session_refs(spec, plan, session_idx)
            P_new, A_tilde_new = build_new_class_prototypes_official(
                backbone, stag_model, full_train_ds, refs, class_list, device, M=Config.num_views,
            )
        else:
            P_new, A_tilde_new = build_new_class_prototypes(
                backbone, stag_model, full_train_ds, class_list, device, shot=shot,
                M=Config.num_views, rng=rng,
            )
        H_new_rows = write_new_memory_rows(stag_model, P_new, A_tilde_new, session_idx, device)

        H = torch.cat([H, H_new_rows], dim=0)
        class_order.extend(class_list)

        novel_mask = torch.tensor([c not in base_set for c in class_order], dtype=torch.bool)
        m = evaluate_cumulative(backbone, stag_model, full_test_ds, class_order, H, device,
                                novel_mask=novel_mask, base_set=base_set)
        acc, acc_base, acc_novel = m["acc"], m["acc_base"], m["acc_novel"]
        hm = (2 * acc_base * acc_novel / (acc_base + acc_novel)
              if (acc_base == acc_base and acc_novel == acc_novel and (acc_base + acc_novel) > 0)
              else float("nan"))
        results.append({
            "session": session_idx, "num_classes_seen": len(class_order),
            "accuracy": acc, "acc_base": acc_base, "acc_novel": acc_novel, "harmonic_mean": hm,
        })

        label = "Session 0 (base)" if session_idx == 0 else f"Session {session_idx} (+{len(class_list)} novel)"
        extra = "" if session_idx == 0 else (f" | A_B={acc_base*100:.2f}% A_N={acc_novel*100:.2f}% "
                                             f"HM={hm*100:.2f}%")
        print(f"[{label}] classes_seen={len(class_order):3d} | cumulative_test_acc={acc*100:.2f}%{extra} "
              f"| {time.time()-t0:.1f}s")

    acc0 = results[0]["accuracy"]
    accT = results[-1]["accuracy"]
    pd = (acc0 - accT) * 100
    avg_acc = float(np.mean([r["accuracy"] for r in results])) * 100
    aB = results[-1]["acc_base"] * 100
    aN = results[-1]["acc_novel"] * 100
    hmT = results[-1]["harmonic_mean"] * 100

    print("\n=== Summary (standard FSCIL metrics) ===")
    print(f"Session 0 accuracy (base only):        {acc0*100:.2f}%")
    print(f"Final session accuracy (all classes):  {accT*100:.2f}%")
    print(f"Performance Drop (PD = Acc0 - AccT):     {pd:.2f} points")
    print(f"Average accuracy across all sessions:  {avg_acc:.2f}%")
    print(f"Final base-only accuracy (A_B):        {aB:.2f}%")
    print(f"Final novel-only accuracy (A_N):       {aN:.2f}%")
    print(f"Final harmonic mean (A_B, A_N):        {hmT:.2f}%")

    os.makedirs(Config.ckpt_dir, exist_ok=True)
    out_path = os.path.join(Config.ckpt_dir, "incremental_results.json")
    with open(out_path, "w") as f:
        json.dump({"dataset": spec.key, "results": results, "PD": pd, "avg_accuracy": avg_acc,
                   "A_B": aB, "A_N": aN, "harmonic_mean": hmT}, f, indent=2)
    print(f"\nSaved results -> {out_path}")


if __name__ == "__main__":
    main()
