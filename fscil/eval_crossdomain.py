"""
Cross-domain FSCIL evaluation: miniImageNet base -> CUB-200 novel.

This is the "forward-compatibility stress test" from the Review-1 deck: a base
session trained on miniImageNet (general objects) is frozen, then novel classes
from CUB-200 (fine-grained birds) are streamed in incrementally. It probes
whether the reserved / spread embedding space generalizes beyond its training
distribution, or only within it.

PROTOCOL (this project's definition -- there is no single standard for
miniImageNet->CUB, so we fix a reproducible one):
  * Backbone + STAG-STI weights: the frozen miniImageNet base-session model
    (checkpoints/miniimagenet/). Trained the normal way via
    `python -m fscil.train_session0 --dataset miniimagenet`.
  * Base session (t=0): miniImageNet's 60 base classes -> global labels 0..59,
    evaluated on miniImageNet test images.
  * Novel sessions (t=1..10): CUB-200 classes, in CUB class-id order, 10 classes
    per session (10-way, matching CUB in-domain), 5-shot support -> global labels
    60..159. Support = the first 5 CUB *train* images per class (deterministic);
    evaluation on CUB *test* images of every CUB class seen so far.
  * All CUB images are resized to 84x84 (the miniImageNet backbone's input) and
    normalized with ImageNet statistics, so the frozen backbone sees in-domain-
    shaped inputs.

Everything is frozen (no gradient updates), exactly like in-domain incremental
eval. Metrics: per-session accuracy, PD, A_B (miniImageNet base), A_N (CUB
novel), harmonic mean.

Run (after the miniImageNet base model exists and CUB data is present):
    python -m fscil.eval_crossdomain
    python -m fscil.eval_crossdomain --novel_sessions 10 --shot 5
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from fscil.config import Config
from fscil.data import build_dataset, base_transforms, load_plan, IndexedDataset
from fscil.datasets import get_spec
from fscil.backbone import build_backbone
from fscil.pipeline import StagStiModel
from fscil.eval_incremental import (
    _prototypes_from_indices, write_new_memory_rows, get_device, set_seed,
)


def build_crossdomain_plan(base_spec, novel_spec, num_novel_sessions):
    """Assign a shared global label space: miniImageNet base -> 0..59,
    CUB novel classes (CUB class-id order) -> 60.. . Returns per-source
    minimal plans (just key_to_global, which is all IndexedDataset needs) plus
    the session -> global-label grouping."""
    base_plan = load_plan(base_spec)
    novel_plan = load_plan(novel_spec)

    base_keys = base_plan["class_order"][:base_spec.num_base_classes]
    base_map = {k: i for i, k in enumerate(base_keys)}                 # 0..59

    novel_way = novel_spec.way
    n_novel = novel_way * num_novel_sessions
    novel_keys = novel_plan["class_order"][:n_novel]                    # CUB id order
    offset = len(base_keys)
    novel_map = {k: offset + i for i, k in enumerate(novel_keys)}       # 60..

    # sessions: base first, then groups of `novel_way`
    sessions = [sorted(base_map.values())]
    for s in range(num_novel_sessions):
        grp = [novel_map[k] for k in novel_keys[s * novel_way:(s + 1) * novel_way]]
        sessions.append(sorted(grp))

    return {
        "base_plan": {"key_to_global": base_map},
        "novel_plan": {"key_to_global": novel_map},
        "session_classes": sessions,
        "num_base": len(base_keys),
    }


@torch.no_grad()
def _prototypes_random(backbone, stag_model, ds, class_list, device, shot, M):
    """Deterministic-first-`shot` support prototypes for the given global
    labels from dataset `ds` (fantasy views are deterministic now)."""
    per_class_indices = []
    for c in class_list:
        idxs = ds.indices_for_class(c)
        per_class_indices.append(idxs[:shot] if len(idxs) >= shot else idxs)
    return _prototypes_from_indices(backbone, stag_model, ds, per_class_indices, device, M)


@torch.no_grad()
def evaluate_crossdomain(backbone, stag_model, class_to_source, class_order, H, device,
                         base_globals, batch_size=128):
    """Cumulative accuracy over class_order, pulling each class's TEST images
    from its own source dataset (miniImageNet for base, CUB for novel).
    class_to_source: {global_label -> (test_dataset, [sample indices])}."""
    pos_map = {c: i for i, c in enumerate(class_order)}
    base_set = set(base_globals)
    # per-CLASS novel mask over the columns of `logits` (aligned to class_order)
    novel_col = torch.tensor([c not in base_set for c in class_order],
                             dtype=torch.float, device=device)
    correct = total = 0
    base_c = base_t = nov_c = nov_t = 0

    # gather (dataset, idx, global_label) triples
    items = []
    for c in class_order:
        ds, idxs = class_to_source[c]
        for i in idxs:
            items.append((ds, i, c))

    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        imgs = torch.stack([ds[i][0] for ds, i, _ in chunk]).to(device)
        true_pos = torch.tensor([pos_map[c] for _, _, c in chunk], device=device)
        feats = backbone(imgs)
        z_q = stag_model.projection(feats)
        logits = stag_model.classify_query(z_q, H)
        if Config.novel_logit_bias != 0.0:
            logits = logits + novel_col * Config.novel_logit_bias  # broadcast over columns
        preds = logits.argmax(1)
        hit = (preds == true_pos)
        correct += hit.sum().item(); total += len(chunk)
        for j, (_, _, c) in enumerate(chunk):
            if c in base_set:
                base_t += 1; base_c += int(hit[j].item())
            else:
                nov_t += 1; nov_c += int(hit[j].item())
    return {
        "acc": correct / total,
        "acc_base": base_c / base_t if base_t else float("nan"),
        "acc_novel": nov_c / nov_t if nov_t else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--novel_sessions", type=int, default=10,
                        help="number of CUB novel sessions (10-way each)")
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--backbone_ckpt", type=str, default=None)
    parser.add_argument("--stag_ckpt", type=str, default=None)
    args = parser.parse_args()

    # operate in miniImageNet mode: 84px input, ImageNet norm, resnet18 backbone
    base_spec = Config.apply_dataset("miniimagenet")
    novel_spec = get_spec("cub200")
    set_seed(Config.seed)
    device = get_device(Config.device)
    print(f"Using device: {device}")
    print(f"Cross-domain: {base_spec.pretty_name} base -> {novel_spec.pretty_name} novel "
          f"({args.novel_sessions} x {novel_spec.way}-way {args.shot}-shot @ {base_spec.image_size}px)")

    backbone_ckpt = args.backbone_ckpt or os.path.join(Config.backbone_ckpt_dir, "backbone_base.pt")
    stag_ckpt = args.stag_ckpt or os.path.join(Config.ckpt_dir, "stag_sti_session0.pt")
    if not (os.path.exists(backbone_ckpt) and os.path.exists(stag_ckpt)):
        raise FileNotFoundError(
            f"Need the miniImageNet base model first:\n  {backbone_ckpt}\n  {stag_ckpt}\n"
            "Run `python -m fscil.train_session0 --dataset miniimagenet`.")

    backbone = build_backbone(Config.backbone_type, out_dim=Config.backbone_out_dim).to(device)
    backbone.load_state_dict(torch.load(backbone_ckpt, map_location=device))
    backbone.freeze()
    stag_model = StagStiModel().to(device)
    stag_model.load_state_dict(torch.load(stag_ckpt, map_location=device))
    stag_model.eval()

    plan = build_crossdomain_plan(base_spec, novel_spec, args.novel_sessions)
    base_globals = plan["session_classes"][0]

    # 84px transform (ImageNet norm) reused for both sources' test images
    test_tf = base_transforms(train=False, spec=base_spec)

    # miniImageNet base: train (for base prototypes) + test (for base eval)
    mini_train = IndexedDataset(base_spec, plan["base_plan"], train=True,
                                allowed_globals=base_globals, transform=None)
    mini_test = IndexedDataset(base_spec, plan["base_plan"], train=False,
                               allowed_globals=base_globals, transform=test_tf)
    # CUB novel: train (support) + test (eval), forced to 84px via test_tf
    cub_train = IndexedDataset(novel_spec, plan["novel_plan"], train=True, transform=None)
    cub_test = IndexedDataset(novel_spec, plan["novel_plan"], train=False, transform=test_tf)

    M = Config.num_views
    H = torch.zeros(0, Config.proj_dim, device=device)
    class_order = []
    class_to_source = {}
    results = []

    for session_idx, class_list in enumerate(plan["session_classes"]):
        t0 = time.time()
        if session_idx == 0:
            P_new, A_new = _prototypes_random(backbone, stag_model, mini_train, class_list,
                                              device, args.shot, M)
            for c in class_list:
                class_to_source[c] = (mini_test, mini_test.indices_for_class(c))
        else:
            P_new, A_new = _prototypes_random(backbone, stag_model, cub_train, class_list,
                                              device, args.shot, M)
            for c in class_list:
                class_to_source[c] = (cub_test, cub_test.indices_for_class(c))

        H_rows = write_new_memory_rows(stag_model, P_new, A_new, session_idx, device)
        H = torch.cat([H, H_rows], dim=0)
        class_order.extend(class_list)

        m = evaluate_crossdomain(backbone, stag_model, class_to_source, class_order, H,
                                 device, base_globals)
        hm = (2 * m["acc_base"] * m["acc_novel"] / (m["acc_base"] + m["acc_novel"])
              if session_idx >= 1 and (m["acc_base"] + m["acc_novel"]) > 0 else float("nan"))
        results.append({"session": session_idx, "num_classes_seen": len(class_order), **m,
                        "harmonic_mean": hm})
        label = "Session 0 (miniImageNet base)" if session_idx == 0 else \
                f"Session {session_idx} (+{len(class_list)} CUB novel)"
        extra = "" if session_idx == 0 else (f" | A_B={m['acc_base']*100:.2f}% "
                                             f"A_N={m['acc_novel']*100:.2f}% HM={hm*100:.2f}%")
        print(f"[{label}] seen={len(class_order):3d} | acc={m['acc']*100:.2f}%{extra} "
              f"| {time.time()-t0:.1f}s")

    acc0, accT = results[0]["acc"], results[-1]["acc"]
    pd = (acc0 - accT) * 100
    print("\n=== Cross-domain summary (miniImageNet -> CUB) ===")
    print(f"Base (miniImageNet) accuracy:  {acc0*100:.2f}%")
    print(f"Final accuracy (all seen):     {accT*100:.2f}%")
    print(f"Performance Drop (PD):         {pd:.2f} points")
    print(f"Final A_B / A_N / HM:          {results[-1]['acc_base']*100:.2f}% / "
          f"{results[-1]['acc_novel']*100:.2f}% / {results[-1]['harmonic_mean']*100:.2f}%")

    out_dir = os.path.join(Config.repo_root, "checkpoints", "crossdomain_mini2cub")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "crossdomain_results.json")
    with open(out_path, "w") as f:
        json.dump({"protocol": "miniImageNet->CUB", "results": results, "PD": pd}, f, indent=2)
    print(f"\nSaved results -> {out_path}")


if __name__ == "__main__":
    main()
