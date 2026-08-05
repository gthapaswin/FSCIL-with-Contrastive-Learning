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
from fscil.data import IndexedCIFAR100, base_transforms, get_base_novel_split, make_fantasy_views
from fscil.backbone import CifarResNet18
from fscil.pipeline import StagStiModel
from fscil.train_session0 import get_device, set_seed

import torchvision


def build_new_class_prototypes(backbone, stag_model, full_train_ds, class_list, device, shot, M, rng):
    """class_list: list of RAW CIFAR labels introduced this session.
    Returns P_new: (way, d'), A_tilde_new: (way, way)."""
    way = len(class_list)
    per_class_views = []
    for c in class_list:
        idxs = full_train_ds.indices_for_class(c)
        chosen = rng.sample(idxs, shot) if len(idxs) >= shot else [rng.choice(idxs) for _ in range(shot)]
        shot_views = []
        for i in chosen:
            raw_img = full_train_ds.data[i]
            pil = torchvision.transforms.functional.to_pil_image(raw_img)
            views = make_fantasy_views(pil, M)          # (M, 3, H, W)
            shot_views.append(views)
        per_class_views.append(torch.stack(shot_views, dim=0))   # (shot, M, 3, H, W)
    support_imgs = torch.stack(per_class_views, dim=0).to(device)  # (way, shot, M, 3, H, W)

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
def evaluate_cumulative(backbone, stag_model, full_test_ds, class_order, H, device, batch_size=256):
    """Accuracy over the test set restricted to every class in class_order,
    classified against the current persistent memory H (rows aligned to
    class_order)."""
    pos_map = {c: i for i, c in enumerate(class_order)}
    indices = []
    for c in class_order:
        indices.extend(full_test_ds.indices_for_class(c))

    correct, total = 0, 0
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        imgs, true_pos = [], []
        for i in batch_idx:
            img, raw_label = full_test_ds[i]
            imgs.append(img)
            true_pos.append(pos_map[raw_label])
        imgs = torch.stack(imgs, dim=0).to(device)
        true_pos = torch.tensor(true_pos, device=device)

        feats = backbone(imgs)
        z_q = stag_model.projection(feats)
        logits = stag_model.classify_query(z_q, H)
        preds = logits.argmax(dim=1)
        correct += (preds == true_pos).sum().item()
        total += len(batch_idx)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, default=Config.shot)
    parser.add_argument("--backbone_ckpt", type=str, default=None,
                         help="Path to backbone checkpoint. Defaults to checkpoints/backbone_base.pt "
                              "(final-epoch). Pass checkpoints/backbone_base_best.pt to use the best-val "
                              "checkpoint instead.")
    parser.add_argument("--stag_ckpt", type=str, default=None,
                         help="Path to STAG-STI checkpoint. Defaults to checkpoints/stag_sti_session0.pt "
                              "(final-epoch). Pass checkpoints/stag_sti_session0_best.pt to use the "
                              "best-val checkpoint instead.")
    args = parser.parse_args()

    set_seed(Config.seed)
    device = get_device(Config.device)
    print(f"Using device: {device}")

    base_classes, incremental_sessions = get_base_novel_split()
    all_sessions = [base_classes] + incremental_sessions

    backbone_ckpt = args.backbone_ckpt or os.path.join(Config.ckpt_dir, "backbone_base.pt")
    stag_ckpt = args.stag_ckpt or os.path.join(Config.ckpt_dir, "stag_sti_session0.pt")
    if not (os.path.exists(backbone_ckpt) and os.path.exists(stag_ckpt)):
        raise FileNotFoundError(
            f"Missing checkpoints. Expected:\n  {backbone_ckpt}\n  {stag_ckpt}\n"
            "Run `python -m fscil.train_session0` first."
        )

    backbone = CifarResNet18(out_dim=Config.backbone_out_dim).to(device)
    backbone.load_state_dict(torch.load(backbone_ckpt, map_location=device))
    backbone.freeze()

    stag_model = StagStiModel().to(device)
    stag_model.load_state_dict(torch.load(stag_ckpt, map_location=device))
    stag_model.eval()

    full_train_ds = IndexedCIFAR100(Config.data_root, train=True, class_subset=list(range(100)),
                                     transform=None, download=True)
    full_test_ds = IndexedCIFAR100(Config.data_root, train=False, class_subset=list(range(100)),
                                    transform=base_transforms(train=False), download=True)

    rng = random.Random(Config.seed + 1000)

    H = torch.zeros(0, Config.proj_dim, device=device)
    class_order = []
    results = []

    print(f"\n=== Incremental FSCIL evaluation: {len(all_sessions)} sessions "
          f"(session 0 = {len(base_classes)} base classes, sessions 1-8 = 5-way {args.shot}-shot) ===")

    for session_idx, class_list in enumerate(all_sessions):
        t0 = time.time()
        P_new, A_tilde_new = build_new_class_prototypes(
            backbone, stag_model, full_train_ds, class_list, device, shot=args.shot,
            M=Config.num_views, rng=rng,
        )
        H_new_rows = write_new_memory_rows(stag_model, P_new, A_tilde_new, session_idx, device)

        H = torch.cat([H, H_new_rows], dim=0)
        class_order.extend(class_list)

        acc = evaluate_cumulative(backbone, stag_model, full_test_ds, class_order, H, device)
        results.append({"session": session_idx, "num_classes_seen": len(class_order), "accuracy": acc})

        label = "Session 0 (base)" if session_idx == 0 else f"Session {session_idx} (+{len(class_list)} novel)"
        print(f"[{label}] classes_seen={len(class_order):3d} | cumulative_test_acc={acc*100:.2f}% "
              f"| {time.time()-t0:.1f}s")

    acc0 = results[0]["accuracy"]
    accT = results[-1]["accuracy"]
    pd = (acc0 - accT) * 100
    avg_acc = float(np.mean([r["accuracy"] for r in results])) * 100

    print("\n=== Summary (standard FSCIL metrics) ===")
    print(f"Session 0 accuracy (base only):        {acc0*100:.2f}%")
    print(f"Final session accuracy (all classes):  {accT*100:.2f}%")
    print(f"Performance Drop (PD = Acc0 - AccT):     {pd:.2f} points")
    print(f"Average accuracy across all sessions:  {avg_acc:.2f}%")

    out_path = os.path.join(Config.ckpt_dir, "incremental_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "PD": pd, "avg_accuracy": avg_acc}, f, indent=2)
    print(f"\nSaved results -> {out_path}")


if __name__ == "__main__":
    main()
