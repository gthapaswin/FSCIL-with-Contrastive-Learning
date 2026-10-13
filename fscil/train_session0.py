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
from fscil.data import (build_dataset, base_transforms, get_official_split, EpisodeSampler,
                        contrastive_transforms, TwoViewDataset, make_fantasy_views)
from fscil.backbone import BackboneWithHead, ReserveHead, build_backbone
from fscil.pipeline import StagStiModel


# ---------------------------------------------------------------------------
# CLOSER Phase-A losses (self-supervised spread + inter-class compactness)
# ---------------------------------------------------------------------------
def nt_xent(z1, z2, temp):
    """SimCLR NT-Xent over two views (z1, z2 are L2-normalized, (N, d))."""
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                     # (2N, d)
    sim = torch.matmul(z, z.T) / temp                    # (2N, 2N)
    sim.fill_diagonal_(float("-inf"))
    targets = (torch.arange(2 * N, device=z.device) + N) % (2 * N)  # positive = other view
    return F.cross_entropy(sim, targets)


def interclass_compactness(feat, labels):
    """Mean pairwise distance between (L2-normalized) class means -- minimizing
    it pulls classes CLOSER, preserving shared features for novel transfer."""
    uniq = labels.unique()
    if uniq.numel() < 2:
        return feat.new_zeros(())
    means = torch.stack([feat[labels == c].mean(0) for c in uniq])
    means = F.normalize(means, dim=-1)
    d = torch.cdist(means, means)
    C = means.size(0)
    return d.sum() / (C * (C - 1))


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
# Resumable training state. Long runs on a laptop get killed partway through;
# saving model + optimizer + LR-scheduler + RNG state every epoch lets a run
# continue on the exact cosine schedule instead of being salvaged half-trained.
# ---------------------------------------------------------------------------
def _rng_state(extra_rngs=()):
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "random": random.getstate(), "extra": [r.getstate() for r in extra_rngs]}


def _set_rng_state(s, extra_rngs=()):
    torch.set_rng_state(s["torch"].cpu())   # map_location may have moved it to the GPU
    np.random.set_state(s["numpy"])
    random.setstate(s["random"])
    for r, st in zip(extra_rngs, s.get("extra", [])):
        r.setstate(st)


def _save_state(path, state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)   # atomic: a kill mid-write never corrupts the last good state


def _load_state(path, epochs, device):
    """Return the saved state, or None. Refuses a state from a run with a
    different epoch count, since its cosine schedule would not match."""
    if not os.path.exists(path):
        return None
    st = torch.load(path, map_location=device, weights_only=False)
    if st["epochs"] != epochs:
        raise RuntimeError(f"{path} was saved for a {st['epochs']}-epoch run, but this run asks for "
                           f"{epochs}. Delete it or pass the original epoch count.")
    return st


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
# Forward-compatible reserve (Phase A, --reserve)
# ---------------------------------------------------------------------------
def reserve_loss(model, imgs, labels, nb):
    """FACT-style loss over num_base real + virtual prototypes. Returns
    (loss, base-only logits for accuracy).
      l_cls  : CE over real + virtual, so base samples stay off the virtual slots
      l_slot : with the true class masked out, the nearest virtual prototype is
               the target -- each base sample keeps a virtual class as its
               runner-up, reserving space next to it for future classes
      l_virt : a 50/50-ish mixture of two different base classes is a stand-in
               for an unseen class, and is trained onto its nearest virtual slot
    """
    B = imgs.size(0)
    lam = float(np.random.beta(Config.reserve_mix_alpha, Config.reserve_mix_alpha))
    perm = torch.randperm(B, device=imgs.device)
    diff = labels != labels[perm]
    mixed = (lam * imgs + (1 - lam) * imgs[perm])[diff]
    # --reserve_cutmix: the real-class branch also gets Phase A's Mixup/CutMix
    # regularization (without it the reserve backbone hit 99% train accuracy)
    if Config.reserve_cutmix:
        real, la, lb, lc = mixup_cutmix_batch(imgs.clone(), labels, nb, imgs.device)
    else:
        real, la, lb, lc = imgs, labels, labels, 1.0
    logits_all, _ = model(torch.cat([real, mixed]))   # one pass: shared BN stats
    logits, logits_m = logits_all[:B], logits_all[B:]

    l_cls = (lc * F.cross_entropy(logits, la, label_smoothing=Config.label_smoothing)
             + (1 - lc) * F.cross_entropy(logits, lb, label_smoothing=Config.label_smoothing))
    masked = logits.scatter(1, la[:, None], -1e4).scatter(1, lb[:, None], -1e4)
    slot = masked[:, nb:].argmax(1) + nb
    l_slot = F.cross_entropy(masked, slot)
    loss = l_cls + Config.reserve_lambda_slot * l_slot
    if logits_m.size(0) > 0:
        vslot = logits_m[:, nb:].argmax(1) + nb
        loss = loss + Config.reserve_lambda_virtual * F.cross_entropy(logits_m, vslot)
    return loss, logits[:, :nb], (la if lc >= 0.5 else lb)


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

    nb = len(base_classes)
    if Config.use_reserve:
        nv = Config.reserve_num_virtual or Config.num_novel_classes
        model = ReserveHead(nb, nv, out_dim=Config.backbone_out_dim, backbone_type=Config.backbone_type,
                            scale=Config.reserve_scale).to(device)
    else:
        model = BackboneWithHead(num_base_classes=nb, out_dim=Config.backbone_out_dim,
                                 backbone_type=Config.backbone_type).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=Config.backbone_pretrain_lr,
                           momentum=0.9, weight_decay=Config.backbone_pretrain_weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"\n=== Phase A: backbone pretraining on {len(base_classes)} base classes for {epochs} epochs ===")
    if Config.use_reserve:
        print(f"Forward-compatible reserve: {nv} virtual prototypes (scale {Config.reserve_scale}, "
              f"lambda_slot {Config.reserve_lambda_slot}, lambda_virtual {Config.reserve_lambda_virtual}, "
              f"Mixup/CutMix on real classes: {Config.reserve_cutmix})")
    elif Config.use_mixup_cutmix:
        print("Mixup/CutMix enabled (randomly applied per batch)")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_epoch, patience_ctr = 0.0, 0, 0
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    best_path = out_path.replace(".pt", "_best.pt")
    state_path = os.path.join(out_dir, "phase_a_state.pt")

    start_epoch = 1
    st = _load_state(state_path, epochs, device)
    if st is not None:
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        history = st["history"]
        best_val_acc, best_epoch, patience_ctr = st["best_val_acc"], st["best_epoch"], st["patience_ctr"]
        _set_rng_state(st["rng"])
        start_epoch = st["epoch"] + 1
        if st["done"]:
            print(f"[Backbone] Phase A already finished ({st['epoch']} epochs, best {best_val_acc*100:.2f}%)")
            start_epoch = epochs + 1
        else:
            print(f"[Backbone] resuming at epoch {start_epoch}/{epochs} (best so far {best_val_acc*100:.2f}%)")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if Config.use_reserve:
                loss, logits, acc_labels = reserve_loss(model, imgs, labels, nb)
            elif Config.use_mixup_cutmix:
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
                logits = logits[:, :nb]   # reserve head: score base classes only
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
              f"| train_acc {train_acc*100:.2f}% | val_acc {val_acc*100:.2f}%{flag} | {time.time()-t0:.1f}s",
              flush=True)

        # patience 0 disables early stopping (long cosine runs plateau mid-schedule)
        stop = Config.early_stop_patience > 0 and patience_ctr >= Config.early_stop_patience
        _save_state(state_path, {
            "epochs": epochs, "epoch": epoch, "done": stop or epoch == epochs,
            "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
            "history": history, "best_val_acc": best_val_acc, "best_epoch": best_epoch,
            "patience_ctr": patience_ctr, "rng": _rng_state(),
        })
        if stop:
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
    if Config.use_heldout:
        R = raw_class_means(raw_feats.view(way, shot, M, -1)[:, :, 0])
        logits = fused_logits(stag_model, z_q, H_t, query_feats, R, Config.residual_lambda)
    else:
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


def raw_class_means(f):
    """Raw-feature residual branch: class mean of L2-normalized backbone
    features of the clean (identity-view) support shots. f: (way, shot, 512)."""
    return F.normalize(F.normalize(f, dim=-1).mean(1), dim=-1)


def fused_logits(stag_model, z_q, H, f_q, R, lam, bias=0.0):
    """inv_tau * ((1-lam) cos(z_q, H) + lam cos(f_q, R)) + bias. lam=0 is
    STAG-STI alone, lam=1 plain nearest-class-mean on raw features. Must match
    eval_incremental._fused_metrics."""
    inv_tau = stag_model.classifier.log_inv_tau.exp()
    s_stag = F.normalize(z_q, dim=-1) @ F.normalize(H, dim=-1).T
    s_raw = F.normalize(f_q, dim=-1) @ F.normalize(R, dim=-1).T
    return inv_tau * ((1 - lam) * s_stag + lam * s_raw) + bias


def split_heldout(base_classes):
    """Deterministic (seeded) split of the base classes into Phase B training
    classes and held-out stand-in novel classes."""
    rng = random.Random(Config.seed + 7)
    held = sorted(rng.sample(list(base_classes), Config.heldout_classes))
    return [c for c in base_classes if c not in held], held


class HeldoutValidator:
    """Phase B model selection on a small FSCIL-style test, scored like
    eval_incremental --full_base: seen (Phase B training) classes get
    abundant-shot prototypes, held-out classes arrive as 5-way 5-shot groups,
    and test queries of all of them are classified jointly with the
    novel-logit bias on the held-out side. HM(seen acc, held-out acc) is
    averaged over several fixed support draws. The backbone is frozen, so all
    backbone features are extracted once here; each call only reruns the head.
    """

    def __init__(self, backbone, seen_classes, held_classes, device):
        self.device = device
        rng = random.Random(Config.seed + 11)
        train_ds = build_dataset(train=True, allowed_globals=seen_classes + held_classes, transform=None)
        test_ds = build_dataset(train=False, allowed_globals=seen_classes + held_classes,
                                transform=base_transforms(train=False))
        M = Config.num_views

        @torch.no_grad()
        def support_feats(idxs):
            views = torch.stack([make_fantasy_views(train_ds.get_pil(i), M) for i in idxs], 0)
            return backbone(views.view(-1, *views.shape[-3:]).to(device)).view(len(idxs), M, -1)

        def pick(c, n):
            idxs = train_ds.indices_for_class(c)
            return rng.sample(idxs, min(n, len(idxs)))

        self.seen = [support_feats(pick(c, Config.heldout_seen_shots)) for c in seen_classes]
        self.held_draws = [[support_feats(pick(c, Config.episode_shot)) for c in held_classes]
                           for _ in range(Config.heldout_draws)]
        q_idx, y = [], []
        for pos, c in enumerate(seen_classes + held_classes):
            idxs = test_ds.indices_for_class(c)
            idxs = rng.sample(idxs, min(Config.heldout_queries, len(idxs)))
            q_idx += idxs; y += [pos] * len(idxs)
        with torch.no_grad():
            self.f_q = torch.cat([backbone(torch.stack([test_ds[i][0] for i in q_idx[s:s + 256]]).to(device))
                                  for s in range(0, len(q_idx), 256)])
        self.y = torch.tensor(y, device=device)
        self.n_seen = len(seen_classes)
        self.bias = torch.zeros(len(seen_classes) + len(held_classes), device=device)
        self.bias[self.n_seen:] = Config.novel_logit_bias

    def _prototypes(self, stag_model, feats, group):
        """feats: list of (shots, M, 512) per class -> H, R; the graph runs in
        groups of `group` classes, as eval chunks its sessions."""
        Hs, Rs = [], []
        dev = self.device
        for st in range(0, len(feats), group):
            g = feats[st:st + group]
            way, M = len(g), g[0].shape[1]
            w_cm = torch.stack([f.mean(0) for f in g])                       # (way, M, 512)
            cid = torch.arange(way, device=dev).unsqueeze(1).expand(way, M).reshape(-1)
            vid = torch.arange(M, device=dev).unsqueeze(0).expand(way, M).reshape(-1)
            _, h_en, A_soft = stag_model.encode_prototypes(w_cm.reshape(way * M, -1), cid, vid)
            P, A_tilde = stag_model.build_prototypes(h_en, A_soft, cid, way)
            H, _, _ = stag_model.update_memory(torch.zeros_like(P), P, A_tilde, torch.zeros(way, device=dev),
                                               session=0, new_class_mask=torch.ones(way, dtype=torch.bool, device=dev))
            Hs.append(H)
            Rs.append(torch.stack([F.normalize(F.normalize(f[:, 0], dim=-1).mean(0), dim=-1) for f in g]))
        return torch.cat(Hs), torch.cat(Rs)

    @torch.no_grad()
    def __call__(self, stag_model, lams):
        """Returns {lam: {"seen", "held", "hm"}} averaged over the support draws."""
        stag_model.eval()
        H_s, R_s = self._prototypes(stag_model, self.seen, Config.episode_way)
        z_q = stag_model.projection(self.f_q)
        is_seen = self.y < self.n_seen
        acc = {lam: [] for lam in lams}
        for draw in self.held_draws:
            H_h, R_h = self._prototypes(stag_model, draw, Config.heldout_val_held_way)
            H, R = torch.cat([H_s, H_h]), torch.cat([R_s, R_h])
            for lam in lams:
                hit = (fused_logits(stag_model, z_q, H, self.f_q, R, lam, self.bias).argmax(1) == self.y).float()
                acc[lam].append((hit[is_seen].mean().item(), hit[~is_seen].mean().item()))
        out = {}
        for lam, v in acc.items():
            a, b = float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))
            out[lam] = {"seen": a, "held": b, "hm": 2 * a * b / (a + b) if a + b > 0 else 0.0}
        return out


def train_stag_sti(backbone, base_classes, device, epochs, out_path):
    if Config.use_heldout:
        train_classes, held_classes = split_heldout(base_classes)
        print(f"Held-out Phase B: training on {len(train_classes)} classes, validating with "
              f"{len(held_classes)} held-out classes {held_classes} (residual lambda {Config.residual_lambda})")
        validator = HeldoutValidator(backbone, train_classes, held_classes, device)
        base_classes = train_classes
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
    state_path = os.path.join(os.path.dirname(out_path), "phase_b_state.pt")
    samplers = (train_sampler.rng, val_sampler.rng)

    start_epoch = 1
    st = _load_state(state_path, epochs, device)
    if st is not None:
        stag_model.load_state_dict(st["model"]); optimizer.load_state_dict(st["opt"])
        scheduler.load_state_dict(st["sched"])
        history = st["history"]
        best_val_acc, best_epoch, patience_ctr = st["best_val_acc"], st["best_epoch"], st["patience_ctr"]
        _set_rng_state(st["rng"], samplers)
        start_epoch = st["epoch"] + 1
        if st["done"]:
            print(f"[STAG-STI] Phase B already finished ({st['epoch']} epochs, best {best_val_acc*100:.2f}%)")
            start_epoch = epochs + 1
        else:
            print(f"[STAG-STI] resuming at epoch {start_epoch}/{epochs} (best so far {best_val_acc*100:.2f}%)")

    for epoch in range(start_epoch, epochs + 1):
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
        ho = ""
        if Config.use_heldout:
            # select on base/novel HM of mixed held-out episodes, at the training lambda
            hv = validator(stag_model, [Config.residual_lambda])[Config.residual_lambda]
            history.setdefault("heldout_hm", []).append(hv["hm"])
            val_acc = hv["hm"]
            ho = f" | heldout seen {hv['seen']*100:.2f}% held {hv['held']*100:.2f}% HM {hv['hm']*100:.2f}%"
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
              f"| train_query_acc {history['train_acc'][-1]*100:.2f}% | val_query_acc {history['val_acc'][-1]*100:.2f}%"
              f"{ho}{flag} | {time.time()-t0:.1f}s", flush=True)

        stop = Config.main_early_stop_patience > 0 and patience_ctr >= Config.main_early_stop_patience
        _save_state(state_path, {
            "epochs": epochs, "epoch": epoch, "done": stop or epoch == epochs,
            "model": stag_model.state_dict(), "opt": optimizer.state_dict(), "sched": scheduler.state_dict(),
            "history": history, "best_val_acc": best_val_acc, "best_epoch": best_epoch,
            "patience_ctr": patience_ctr, "rng": _rng_state(samplers),
        })
        if stop:
            print(f"[STAG-STI] early stopping: no val_query_acc improvement for "
                  f"{Config.main_early_stop_patience} epochs (best {best_val_acc*100:.2f}% at epoch {best_epoch})")
            break

    torch.save(stag_model.state_dict(), out_path)
    if Config.use_heldout:
        # The checkpoint and lam are both chosen on held-out classes, never on test.
        best_model = StagStiModel().to(device)
        best_model.load_state_dict(torch.load(best_path, map_location=device))
        sweep = validator(best_model, list(Config.heldout_lams))
        lam_best = max(sweep, key=lambda l: sweep[l]["hm"])
        for lam, v in sweep.items():
            print(f"[held-out] lam={lam:g} seen {v['seen']*100:.2f}% held {v['held']*100:.2f}% "
                  f"HM {v['hm']*100:.2f}%{' <- selected' if lam == lam_best else ''}")
        with open(os.path.join(os.path.dirname(out_path), "heldout_val.json"), "w") as f:
            json.dump({"held_classes": held_classes, "best_epoch": best_epoch, "selected_lambda": lam_best,
                       "sweep": {str(k): v for k, v in sweep.items()}}, f, indent=2)
    print(f"\nSaved final-epoch STAG-STI checkpoint -> {out_path}")
    print(f"Saved best-val-acc STAG-STI checkpoint ({best_val_acc*100:.2f}% @ epoch {best_epoch}) -> {best_path}")
    return stag_model, history


def pretrain_backbone_closer(base_classes, device, epochs, out_path):
    """CLOSER-style Phase A: CE + self-supervised NT-Xent (spread) + inter-class
    compactness (transfer). Two augmented views per image."""
    two_tf = contrastive_transforms()
    base_train = build_dataset(train=True, allowed_globals=base_classes, transform=None)
    train_ds = TwoViewDataset(base_train, two_tf)
    test_ds = build_dataset(train=False, allowed_globals=base_classes,
                            transform=base_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=Config.backbone_pretrain_batch_size,
                              shuffle=True, num_workers=Config.num_workers, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=Config.num_workers)

    model = BackboneWithHead(num_base_classes=len(base_classes), out_dim=Config.backbone_out_dim,
                             backbone_type=Config.backbone_type, ssl_dim=Config.closer_ssl_dim).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=Config.backbone_pretrain_lr, momentum=0.9,
                          weight_decay=Config.backbone_pretrain_weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    print(f"\n=== Phase A (CLOSER) on {len(base_classes)} base classes for {epochs} epochs "
          f"(lambda_ssl={Config.closer_lambda_ssl}, lambda_close={Config.closer_lambda_close}) ===")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_epoch, patience = 0.0, 0, 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    best_path = out_path.replace(".pt", "_best.pt")

    for epoch in range(1, epochs + 1):
        model.train(); t0 = time.time()
        tot, cor, seen = 0.0, 0, 0
        for v1, v2, labels in train_loader:
            v1, v2, labels = v1.to(device), v2.to(device), labels.to(device)
            logits1, feat1 = model(v1)
            _, feat2 = model(v2)
            l_ce = F.cross_entropy(logits1, labels, label_smoothing=Config.label_smoothing)
            l_ssl = nt_xent(model.project_ssl(feat1), model.project_ssl(feat2), Config.closer_ssl_temp)
            l_close = interclass_compactness(feat1, labels)
            loss = l_ce + Config.closer_lambda_ssl * l_ssl + Config.closer_lambda_close * l_close
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * v1.size(0); cor += (logits1.argmax(1) == labels).sum().item(); seen += v1.size(0)
        sched.step()
        train_loss, train_acc = tot / seen, cor / seen

        model.eval(); vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, _ = model(imgs)
                vl += F.cross_entropy(logits, labels).item() * imgs.size(0)
                vc += (logits.argmax(1) == labels).sum().item(); vt += imgs.size(0)
        val_loss, val_acc = vl / vt, vc / vt
        for k, v in [("train_loss", train_loss), ("val_loss", val_loss),
                     ("train_acc", train_acc), ("val_acc", val_acc)]:
            history[k].append(v)
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc, best_epoch, patience = val_acc, epoch, 0
            torch.save(model.backbone.state_dict(), best_path)
        else:
            patience += 1
        print(f"[CLOSER] epoch {epoch:03d}/{epochs} | loss {train_loss:.4f} (val {val_loss:.4f}) "
              f"| train_acc {train_acc*100:.2f}% | val_acc {val_acc*100:.2f}%{' *' if improved else ''} "
              f"| {time.time()-t0:.1f}s")
        if patience >= Config.early_stop_patience:
            print(f"[CLOSER] early stop (best {best_val_acc*100:.2f}% @ {best_epoch})"); break

    torch.save(model.backbone.state_dict(), out_path)
    print(f"Saved CLOSER backbone -> {out_path} (best {best_val_acc*100:.2f}% @ {best_epoch})")
    return model.backbone, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar100",
                         help="cifar100 | miniimagenet | cub200")
    parser.add_argument("--closer", action="store_true",
                         help="Use the CLOSER-style Phase-A objective (transfer-friendly base).")
    parser.add_argument("--reserve", action="store_true",
                         help="Forward-compatible reserve in Phase A (FACT-style virtual prototypes). "
                              "Defaults the run to --tag reserve so the baseline is never overwritten.")
    parser.add_argument("--reserve_cutmix", action="store_true",
                         help="With --reserve, keep Phase A's Mixup/CutMix on the real-class branch.")
    parser.add_argument("--heldout", action="store_true",
                         help="Held-out Phase B: train on base minus Config.heldout_classes classes, select the "
                              "checkpoint and fusion lambda on mixed seen/held-out episodes, residual training.")
    parser.add_argument("--backbone_tag", type=str, default=None,
                         help="Load the frozen backbone from checkpoints/<ds>/<backbone_tag>/backbone_base.pt "
                              "(skips Phase A) while writing Phase B under --tag.")
    parser.add_argument("--tag", type=str, default=None,
                         help="Namespace this run under checkpoints/<ds>/<tag>/ (for variant runs).")
    parser.add_argument("--closer_lambda_close", type=float, default=None,
                         help="Override inter-class compactness weight (default 0.1). "
                              "Non-default values namespace the run as closer_lc<val>/.")
    parser.add_argument("--closer_lambda_ssl", type=float, default=None,
                         help="Override self-supervised NT-Xent weight (default 0.5).")
    parser.add_argument("--ablate", nargs="+", default=None,
                         choices=["none", "supcon", "topology", "agedecay"],
                         help="Disable STAG-STI components for the ablation study "
                              "(namespaces output under checkpoints/<ds>/ablate-...).")
    parser.add_argument("--backbone_epochs", type=int, default=Config.backbone_pretrain_epochs)
    parser.add_argument("--main_epochs", type=int, default=Config.main_epochs)
    parser.add_argument("--skip_backbone_pretrain", action="store_true",
                         help="Skip Phase A and load an existing checkpoint instead.")
    parser.add_argument("--backbone", type=str, default=None, choices=["resnet20"],
                         help="Override the dataset's backbone (resnet20: the 64-d CIFAR ResNet-20 used by "
                              "most FSCIL papers). Pair it with a distinct --tag.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed (default Config.seed=42). A non-default seed namespaces the "
                              "run under seed<N>/ so seed-averaged runs never overwrite the baseline.")
    parser.add_argument("--patience", type=int, default=None,
                         help="Phase-A early-stop patience; 0 disables (recommended for long cosine runs).")
    parser.add_argument("--main_patience", type=int, default=None,
                         help="Phase-B early-stop patience; 0 disables.")
    args = parser.parse_args()
    if args.patience is not None:
        Config.early_stop_patience = args.patience
    if args.main_patience is not None:
        Config.main_early_stop_patience = args.main_patience

    spec = Config.apply_dataset(args.dataset)
    if args.backbone == "resnet20":
        Config.backbone_type, Config.backbone_out_dim = "cifar_resnet20", 64
    ds_dir = Config.ckpt_dir   # checkpoints/<ds>, before any namespacing
    ablations = Config.apply_ablation(args.ablate)
    if args.closer:
        # namespace the whole run so the CLOSER variant never clobbers baseline
        Config.use_closer = True
        if args.closer_lambda_close is not None:
            Config.closer_lambda_close = args.closer_lambda_close
        if args.closer_lambda_ssl is not None:
            Config.closer_lambda_ssl = args.closer_lambda_ssl
        sub = "closer" if Config.closer_lambda_close == 0.1 else f"closer_lc{Config.closer_lambda_close}"
        Config.ckpt_dir = os.path.join(Config.ckpt_dir, sub)
        Config.backbone_ckpt_dir = os.path.join(Config.backbone_ckpt_dir, sub)
    if args.reserve:
        Config.use_reserve = True
        Config.reserve_cutmix = args.reserve_cutmix
        args.tag = args.tag or "reserve"
    if args.tag:
        Config.ckpt_dir = os.path.join(Config.ckpt_dir, args.tag)
        Config.backbone_ckpt_dir = os.path.join(Config.backbone_ckpt_dir, args.tag)
    if args.heldout:
        Config.use_heldout = True
    if args.backbone_tag:
        Config.backbone_ckpt_dir = os.path.join(ds_dir, args.backbone_tag)
        args.skip_backbone_pretrain = True
    if args.seed is not None and args.seed != Config.seed:
        Config.seed = args.seed
        Config.ckpt_dir = os.path.join(Config.ckpt_dir, f"seed{args.seed}")
        Config.backbone_ckpt_dir = os.path.join(Config.backbone_ckpt_dir, f"seed{args.seed}")
    set_seed(Config.seed)
    device = get_device(Config.device)
    print(f"Using device: {device} | seed {Config.seed}")
    print(f"Dataset: {spec.pretty_name} | backbone: {Config.backbone_type} | "
          f"{spec.num_base_classes} base + {spec.num_incremental_sessions}x{spec.way}-way "
          f"{spec.shot}-shot | output -> {Config.ckpt_dir}")
    if ablations:
        print(f"ABLATION: disabled {ablations} "
              f"(lambda_supcon={Config.lambda_supcon}, lambda_graph={Config.lambda_graph})")

    base_classes, incremental_sessions = get_official_split(spec)
    print(f"Base classes ({len(base_classes)}): {base_classes}")
    print(f"Incremental sessions ({spec.num_incremental_sessions} x {spec.way}-way): {incremental_sessions}")

    # backbone lives at the dataset level (shared across ablations); the
    # STAG-STI checkpoint + history go under the (possibly ablated) ckpt_dir.
    os.makedirs(Config.backbone_ckpt_dir, exist_ok=True)
    backbone_ckpt = os.path.join(Config.backbone_ckpt_dir, "backbone_base.pt")
    full_history = {}

    if args.backbone_tag and not os.path.exists(backbone_ckpt):
        raise FileNotFoundError(f"--backbone_tag: no backbone at {backbone_ckpt}")
    if args.skip_backbone_pretrain and os.path.exists(backbone_ckpt):
        backbone = build_backbone(Config.backbone_type, out_dim=Config.backbone_out_dim).to(device)
        backbone.load_state_dict(torch.load(backbone_ckpt, map_location=device))
        print(f"Loaded existing backbone checkpoint from {backbone_ckpt}")
    elif args.closer:
        backbone, phase_a_history = pretrain_backbone_closer(base_classes, device, args.backbone_epochs, backbone_ckpt)
        full_history["phase_a"] = phase_a_history
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
