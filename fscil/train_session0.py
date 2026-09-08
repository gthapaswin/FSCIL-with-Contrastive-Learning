"""
Session 0 (base session) training for STAG-STI on CIFAR-100.

Two phases, both run by this one script:
  Phase A — supervised pretraining of the ResNet18 backbone on the 60 base
            classes (standard cross-entropy). This is what gives the
            "frozen backbone" in Stage 2 something useful to extract.
  Phase B — the actual STAG-STI pipeline: episodic multi-view fantasy
            sampling -> frozen backbone -> projection -> SupCon head ->
            topology scorer -> GATv2 -> prototype pooling -> STI memory ->
            cosine classifier. This is the "50 epochs" your guide asked for
            (--main_epochs, default 50).

OVERFITTING-REDUCTION CHANGES (added after the first two training runs
showed ~99% train vs ~76-84% val accuracy in both phases):
  - Increased weight decay (Phase A: 5e-4 -> 1e-3; Phase B: 5e-4 -> 1e-2)
  - Label smoothing (0.1) on both phases' cross-entropy losses
  - Dropout added inside the Phase A classifier head, the topology scorer,
    and the GATv2 aggregation (see backbone.py / modules.py)
  - RandomErasing (cutout-style) augmentation added to training transforms
  - Val loss is now tracked (not just val accuracy), for both phases
  - Best-val-accuracy checkpoints are saved separately (*_best.pt) alongside
    the final-epoch checkpoint, and early stopping halts a phase once val
    accuracy stops improving for `early_stop_patience` epochs
  - Full per-epoch history (train/val loss & accuracy, all loss components)
    is saved to checkpoints/training_history.json for the plotting script

Run:
    python -m fscil.train_session0
    python -m fscil.train_session0 --main_epochs 50 --backbone_epochs 30
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fscil.config import Config
from fscil.data import build_dataset, base_transforms, get_official_split, EpisodeSampler
from fscil.backbone import BackboneWithHead, build_backbone
from fscil.pipeline import StagStiModel


def get_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Mixup / CutMix (Phase A regularization -- randomly picks one per batch)
# ---------------------------------------------------------------------------
def mixup_cutmix_batch(imgs, labels, num_classes, device):
    """Returns (mixed_imgs, labels_a, labels_b, lam). Loss should be computed
    as lam * CE(logits, labels_a) + (1-lam) * CE(logits, labels_b)."""
    use_cutmix = random.random() < 0.5
    alpha = Config.cutmix_alpha if use_cutmix else Config.mixup_alpha
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(imgs.size(0), device=device)
    labels_b = labels[perm]

    if use_cutmix:
        B, C, H, W = imgs.shape
        cut_ratio = (1 - lam) ** 0.5
        cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
        cy, cx = random.randint(0, H - 1), random.randint(0, W - 1)
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
        imgs[:, :, y1:y2, x1:x2] = imgs[perm][:, :, y1:y2, x1:x2]
        lam = 1 - ((y2 - y1) * (x2 - x1) / (H * W))  # recompute exact lambda from actual patch area
        mixed = imgs
    else:
        mixed = lam * imgs + (1 - lam) * imgs[perm]

    return mixed, labels, labels_b, lam


# ---------------------------------------------------------------------------
# Phase A: backbone pretraining on base classes (plain classification)
# ---------------------------------------------------------------------------
def pretrain_backbone(base_classes, device, epochs, out_path):
    train_ds = build_dataset(train=True, allowed_globals=base_classes,
                             transform=base_transforms(train=True))
    test_ds = build_dataset(train=False, allowed_globals=base_classes,
                            transform=base_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=Config.backbone_pretrain_batch_size,
                               shuffle=True, num_workers=Config.num_workers, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                              num_workers=Config.num_workers)

    model = BackboneWithHead(num_base_classes=len(base_classes), out_dim=Config.backbone_out_dim,
                             backbone_type=Config.backbone_type).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=Config.backbone_pretrain_lr,
                           momentum=0.9, weight_decay=Config.backbone_pretrain_weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"\n=== Phase A: backbone pretraining on {len(base_classes)} base classes for {epochs} epochs ===")
    if Config.use_mixup_cutmix:
        print("Mixup/CutMix enabled (randomly applied per batch)")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_epoch, patience_ctr = 0.0, 0, 0
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    best_path = out_path.replace(".pt", "_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if Config.use_mixup_cutmix:
                imgs, labels_a, labels_b, lam = mixup_cutmix_batch(imgs, labels, len(base_classes), device)
                logits, _ = model(imgs)
                loss = (lam * F.cross_entropy(logits, labels_a, label_smoothing=Config.label_smoothing)
                        + (1 - lam) * F.cross_entropy(logits, labels_b, label_smoothing=Config.label_smoothing))
                # train_acc under mixup is an approximation (checked against the dominant label)
                acc_labels = labels_a if lam >= 0.5 else labels_b
            else:
                logits, _ = model(imgs)
                loss = F.cross_entropy(logits, labels, label_smoothing=Config.label_smoothing)
                acc_labels = labels
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == acc_labels).sum().item()
            total += imgs.size(0)
        sched.step()
        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, _ = model(imgs)
                vloss = F.cross_entropy(logits, labels, label_smoothing=Config.label_smoothing)
                val_loss_sum += vloss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc, best_epoch, patience_ctr = val_acc, epoch, 0
            torch.save(model.backbone.state_dict(), best_path)
        else:
            patience_ctr += 1

        flag = " *" if improved else ""
        print(f"[Backbone] epoch {epoch:03d}/{epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
              f"| train_acc {train_acc*100:.2f}% | val_acc {val_acc*100:.2f}%{flag} | {time.time()-t0:.1f}s")

        if patience_ctr >= Config.early_stop_patience:
            print(f"[Backbone] early stopping: no val_acc improvement for {Config.early_stop_patience} epochs "
                  f"(best {best_val_acc*100:.2f}% at epoch {best_epoch})")
            break

    torch.save(model.backbone.state_dict(), out_path)
    print(f"Saved final-epoch backbone -> {out_path}")
    print(f"Saved best-val-acc backbone ({best_val_acc*100:.2f}% @ epoch {best_epoch}) -> {best_path}")

    return model.backbone, history


# ---------------------------------------------------------------------------
# Phase B: episodic STAG-STI training (the "50 epochs")
# ---------------------------------------------------------------------------
def run_episode(backbone, stag_model, sampler, device, optimizer=None):
    """Runs one episode end-to-end. If optimizer is given, does a training
    step; otherwise runs in eval mode and just returns metrics.

    Stage order (per the corrected pipeline table):
      Stage 1/2: fantasy views -> frozen backbone -> raw per-shot features
      Stage 3.5: view-preserving SupCon on the RAW per-shot features
      Stage 3:   shot-averaging -> w_{c,m} prototypes
      Stage 4/5: W_proj -> topology scorer -> GATv2
      Stage 6:   view pooling -> class prototypes P_t
      Stage 7:   STI memory update -> H_t  (+ L_graph, L_stab)
      Stage 8:   cosine classification of queries against H_t  (+ L_cls)
    """
    train_mode = optimizer is not None
    stag_model.train(train_mode)

    support_imgs, support_labels, query_imgs, query_labels = sampler.sample_episode()
    # support_imgs: (N, M, 3, H, W) where N = way*shot
    N, M, C, H, W = support_imgs.shape
    support_imgs = support_imgs.to(device)
    support_labels = support_labels.to(device)
    query_imgs = query_imgs.to(device)
    query_labels = query_labels.to(device)

    way = int(support_labels.max().item()) + 1
    shot = N // way

    with torch.no_grad():  # backbone always frozen
        flat = support_imgs.view(N * M, C, H, W)
        raw_feats = backbone(flat).view(N, M, -1)        # (N, M, 512) -- z_{i,m}
        query_feats = backbone(query_imgs)                  # (way*query, 512)

    # ---- Stage 3.5: view-preserving SupCon on RAW per-shot features ----
    raw_flat = raw_feats.reshape(N * M, -1)                                             # (N*M, 512)
    raw_class_ids = support_labels.unsqueeze(1).expand(N, M).reshape(-1)                  # (N*M,)
    raw_view_ids = torch.arange(M, device=device).unsqueeze(0).expand(N, M).reshape(-1)     # (N*M,)
    l_supcon = stag_model.supcon_on_raw_features(raw_flat, raw_class_ids, raw_view_ids)

    # ---- Stage 3: shot-averaging -> w_{c,m} ----
    w_cm = raw_feats.view(way, shot, M, -1).mean(dim=1)         # (way, M, 512)
    class_ids = torch.arange(way, device=device).unsqueeze(1).expand(way, M).reshape(-1)  # (way*M,)
    view_ids = torch.arange(M, device=device).unsqueeze(0).expand(way, M).reshape(-1)        # (way*M,)
    node_feats = w_cm.reshape(way * M, -1)                                                    # (K,512)

    # ---- Stage 4/5: projection -> topology -> GATv2 ----
    h0, h_enriched, A_soft = stag_model.encode_prototypes(node_feats, class_ids, view_ids)
    l_graph = stag_model.graph_loss(A_soft, class_ids)

    # ---- Stage 6: pooling ----
    P_t, A_tilde = stag_model.build_prototypes(h_enriched, A_soft, class_ids, way)

    # ---- Stage 7: STI memory update ----
    # Fresh episode -> every class is "new" this session, no persistent memory yet.
    H_prev = torch.zeros_like(P_t)
    class_ages = torch.zeros(way, device=device)
    new_class_mask = torch.ones(way, dtype=torch.bool, device=device)
    H_t, R_gate, phi_p = stag_model.update_memory(H_prev, P_t, A_tilde, class_ages, session=0,
                                                     new_class_mask=new_class_mask)
    l_stab = stag_model.stability_loss(P_t, phi_p, R_gate)

    # ---- Stage 8: cosine classification of queries ----
    z_q = stag_model.projection(query_feats)
    logits = stag_model.classify_query(z_q, H_t)
    l_cls = F.cross_entropy(logits, query_labels, label_smoothing=Config.label_smoothing)

    loss = (Config.lambda_cls * l_cls
            + Config.lambda_supcon * l_supcon
            + Config.lambda_graph * l_graph
            + Config.lambda_stab * l_stab)

    if train_mode:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = (logits.argmax(1) == query_labels).float().mean().item()
    return {
        "loss": loss.item(), "supcon": l_supcon.item(), "graph": l_graph.item(),
        "stab": l_stab.item(), "cls": l_cls.item(), "acc": acc,
    }


def train_stag_sti(backbone, base_classes, device, epochs, out_path):
    train_ds = build_dataset(train=True, allowed_globals=base_classes,
                             transform=base_transforms(train=True))
    val_ds = build_dataset(train=False, allowed_globals=base_classes,
                           transform=base_transforms(train=False))

    train_sampler = EpisodeSampler(train_ds, way=Config.episode_way, shot=Config.episode_shot,
                                    query=Config.episode_query, seed=Config.seed)
    val_sampler = EpisodeSampler(val_ds, way=Config.episode_way, shot=Config.episode_shot,
                                  query=Config.episode_query, seed=Config.seed + 1)

    stag_model = StagStiModel().to(device)
    optimizer = torch.optim.AdamW(stag_model.parameters(), lr=Config.main_lr,
                                   weight_decay=Config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n=== Phase B: STAG-STI episodic training for {epochs} epochs "
          f"({Config.episodes_per_epoch} episodes/epoch, {Config.episode_way}-way "
          f"{Config.episode_shot}-shot) ===")

    keys = ["loss", "supcon", "graph", "stab", "cls", "acc"]
    history = {f"train_{k}": [] for k in keys}
    history.update({f"val_{k}": [] for k in keys})

    best_val_acc, best_epoch, patience_ctr = 0.0, 0, 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    best_path = out_path.replace(".pt", "_best.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        agg = {k: [] for k in keys}
        for _ in range(Config.episodes_per_epoch):
            m = run_episode(backbone, stag_model, train_sampler, device, optimizer)
            for k in keys:
                agg[k].append(m[k])
        scheduler.step()

        val_agg = {k: [] for k in keys}
        with torch.no_grad():
            for _ in range(20):
                vm = run_episode(backbone, stag_model, val_sampler, device, optimizer=None)
                for k in keys:
                    val_agg[k].append(vm[k])

        for k in keys:
            history[f"train_{k}"].append(float(np.mean(agg[k])))
            history[f"val_{k}"].append(float(np.mean(val_agg[k])))

        val_acc = history["val_acc"][-1]
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc, best_epoch, patience_ctr = val_acc, epoch, 0
            torch.save(stag_model.state_dict(), best_path)
        else:
            patience_ctr += 1

        flag = " *" if improved else ""
        print(f"[STAG-STI] epoch {epoch:03d}/{epochs} | "
              f"loss {history['train_loss'][-1]:.4f} (val {history['val_loss'][-1]:.4f}) "
              f"(cls {history['train_cls'][-1]:.4f}, supcon {history['train_supcon'][-1]:.4f}, "
              f"graph {history['train_graph'][-1]:.4f}, stab {history['train_stab'][-1]:.4f}) "
              f"| train_query_acc {history['train_acc'][-1]*100:.2f}% | val_query_acc {val_acc*100:.2f}%{flag} "
              f"| {time.time()-t0:.1f}s")

        if patience_ctr >= Config.main_early_stop_patience:
            print(f"[STAG-STI] early stopping: no val_query_acc improvement for "
                  f"{Config.main_early_stop_patience} epochs (best {best_val_acc*100:.2f}% at epoch {best_epoch})")
            break

    torch.save(stag_model.state_dict(), out_path)
    print(f"\nSaved final-epoch STAG-STI checkpoint -> {out_path}")
    print(f"Saved best-val-acc STAG-STI checkpoint ({best_val_acc*100:.2f}% @ epoch {best_epoch}) -> {best_path}")
    return stag_model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar100",
                         help="cifar100 | miniimagenet | cub200")
    parser.add_argument("--backbone_epochs", type=int, default=Config.backbone_pretrain_epochs)
    parser.add_argument("--main_epochs", type=int, default=Config.main_epochs)
    parser.add_argument("--skip_backbone_pretrain", action="store_true",
                         help="Skip Phase A and load an existing checkpoint instead.")
    args = parser.parse_args()

    spec = Config.apply_dataset(args.dataset)
    set_seed(Config.seed)
    device = get_device(Config.device)
    print(f"Using device: {device}")
    print(f"Dataset: {spec.pretty_name} | backbone: {Config.backbone_type} | "
          f"{spec.num_base_classes} base + {spec.num_incremental_sessions}x{spec.way}-way "
          f"{spec.shot}-shot | output -> {Config.ckpt_dir}")

    base_classes, incremental_sessions = get_official_split(spec)
    print(f"Base classes ({len(base_classes)}): {base_classes}")
    print(f"Incremental sessions ({spec.num_incremental_sessions} x {spec.way}-way): {incremental_sessions}")

    backbone_ckpt = os.path.join(Config.ckpt_dir, "backbone_base.pt")
    full_history = {}

    if args.skip_backbone_pretrain and os.path.exists(backbone_ckpt):
        backbone = build_backbone(Config.backbone_type, out_dim=Config.backbone_out_dim).to(device)
        backbone.load_state_dict(torch.load(backbone_ckpt, map_location=device))
        print(f"Loaded existing backbone checkpoint from {backbone_ckpt}")
    else:
        backbone, phase_a_history = pretrain_backbone(base_classes, device, args.backbone_epochs, backbone_ckpt)
        full_history["phase_a"] = phase_a_history

    backbone.freeze()  # frozen for all of Phase B and every incremental session

    stag_ckpt = os.path.join(Config.ckpt_dir, "stag_sti_session0.pt")
    stag_model, phase_b_history = train_stag_sti(backbone, base_classes, device, args.main_epochs, stag_ckpt)
    full_history["phase_b"] = phase_b_history

    os.makedirs(Config.ckpt_dir, exist_ok=True)
    history_path = os.path.join(Config.ckpt_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(full_history, f, indent=2)
    print(f"\nSaved training history -> {history_path}")
    print("Run `python -m fscil.plotting` to generate comparison plots from this history.")


if __name__ == "__main__":
    main()
