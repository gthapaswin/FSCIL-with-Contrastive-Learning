"""Live presentation demo (CIFAR-100, about 4 minutes). Three acts, each
waits for Enter so the presenter controls the pace:

  1. The pipeline   -- a novel support image, its fantasy views, the learned
                       class topology, and the 8 incremental sessions run live
                       (STAG-STI residual head vs nearest class mean).
  2. Predictions    -- real test images of novel classes, classified three ways.
  3. The finding    -- NCM's offset trade-off curve, where the head lands on
                       it, prediction agreement and score spread.

    .venv/bin/python scripts/demo.py            # presenter mode (Enter to advance)
    .venv/bin/python scripts/demo.py --auto     # run straight through (rehearsal)

The first run caches base prototypes and test features (about 1 min);
later runs start in a few seconds. Figures open in Preview from demo/.
Uses the same code paths and checkpoints as the paper (seed 42,
reserve_cm100 backbone, held-out residual head), so the numbers match it.
"""
import argparse
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fscil.config import Config                                          # noqa: E402
from fscil.data import build_dataset, base_transforms, get_official_split, make_fantasy_views, load_plan  # noqa: E402
from fscil.datasets import official_session_refs                         # noqa: E402
from fscil.backbone import build_backbone                                # noqa: E402
from fscil.pipeline import StagStiModel                                  # noqa: E402
from fscil.train_session0 import get_device, set_seed                    # noqa: E402
from fscil.eval_incremental import (build_base_prototypes, build_new_class_prototypes_official,  # noqa: E402
                                    write_new_memory_rows)

OUT = os.path.join(ROOT, "demo")
B, DIM, BOLD, GRN, RED, YEL, CYN, RST = "\033[1m", "\033[2m", "\033[1m", "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m"
LAM, B0 = 0.5, 0.5          # residual weight and default novel offset used in the paper
AUTO = False


def say(msg="", delay=0.0):
    print(msg, flush=True)
    if delay and not AUTO:
        time.sleep(delay)


def pause(what):
    if AUTO:
        say(f"{DIM}--- {what} ---{RST}")
        return
    input(f"\n{DIM}[Enter] {what} {RST}")


def banner(title):
    say(f"\n{CYN}{B}{'=' * 72}\n  {title}\n{'=' * 72}{RST}")


def show(path):
    if not AUTO and sys.platform == "darwin":
        subprocess.Popen(["open", path])


def hm(a, b):
    return 2 * a * b / (a + b) if a + b else 0.0


def bar(v, width=30, ch="#"):
    return ch * int(round(v / 100 * width))


# --------------------------------------------------------------------------
def setup():
    spec = Config.apply_dataset("cifar100")
    set_seed(Config.seed)
    device = get_device(Config.device)
    plan = load_plan()
    base_classes, sessions = get_official_split(plan=plan)
    inv = {g: k for k, g in plan["key_to_global"].items()}
    names = torchvision.datasets.CIFAR100(root=Config.data_root, train=False, download=False).classes
    name = lambda g: names[int(inv[g])].replace("_", " ")

    ck = os.path.join(ROOT, "checkpoints", "cifar100")
    backbone = build_backbone(Config.backbone_type, out_dim=Config.backbone_out_dim).to(device)
    backbone.load_state_dict(torch.load(os.path.join(ck, "reserve_cm100", "backbone_base.pt"), map_location=device))
    backbone.freeze()
    stag = StagStiModel().to(device)
    stag.load_state_dict(torch.load(os.path.join(ck, "reserve_cm100_heldout", "stag_sti_session0_best.pt"),
                                    map_location=device))
    stag.eval()
    train_ds = build_dataset(train=True, transform=None, plan=plan)
    test_ds = build_dataset(train=False, transform=base_transforms(train=False), plan=plan)
    return dict(spec=spec, device=device, plan=plan, base=base_classes, sessions=sessions, name=name,
                backbone=backbone, stag=stag, train=train_ds, test=test_ds)


@torch.no_grad()
def cached(S):
    """Base prototypes and test features: computed once, then loaded."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "cache_cifar100.pt")
    if os.path.exists(path):
        return torch.load(path, map_location=S["device"])
    say(f"{DIM}First run: building the cache (base prototypes from 100 images x 60 classes x 4 views,"
        f" and features of the 10,000 test images)...{RST}")
    rng = random.Random(Config.seed + 1000)        # same draw as the paper's evaluation
    P, A, R = build_base_prototypes(S["backbone"], S["stag"], S["train"], S["base"], S["device"],
                                    M=Config.num_views, cap=Config.base_proto_cap, rng=rng)
    f, y = [], []
    loader = torch.utils.data.DataLoader(S["test"], batch_size=256)
    for x, t in loader:
        f.append(S["backbone"](x.to(S["device"])))
        y.append(t)
    c = dict(P=P, A=A, R=R, f=torch.cat(f), y=torch.cat(y).to(S["device"]))
    torch.save(c, path)
    return c


# --------------------------------------------------------------------------
@torch.no_grad()
def act1(S, C):
    banner("ACT 1  The STAG-STI pipeline on CIFAR-100")
    nb = sum(p.numel() for p in S["backbone"].parameters()) / 1e6
    nh = sum(p.numel() for p in S["stag"].parameters()) / 1e6
    say(f"  Frozen backbone : ResNet-18, {nb:.1f}M parameters, trained once on the 60 base classes")
    say(f"  STAG-STI head   : {nh:.2f}M parameters (projection, topology scorer, GATv2, STI memory)")
    say(f"  Protocol        : 60 base classes, then 8 sessions x 5 new classes x 5 images each")
    pause("show a few-shot support image and its fantasy views")

    # one support image of the first novel class, and its M deterministic views
    refs = official_session_refs(S["spec"], S["plan"], 1)
    idxs = [S["train"].index_for_ref(r) for r in refs]
    idx = next((i for i in idxs if S["name"](S["train"].targets[i]) == "porcupine"), idxs[0])
    g = S["train"].targets[idx]
    pil = S["train"].get_pil(idx)
    views = make_fantasy_views(pil, Config.num_views)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mean, std = torch.tensor(Config.data_mean)[:, None, None], torch.tensor(Config.data_std)[:, None, None]
    fig, ax = plt.subplots(1, Config.num_views + 1, figsize=(2.2 * (Config.num_views + 1), 2.6))
    ax[0].imshow(pil.resize((128, 128), resample=0)); ax[0].set_title(f"support: {S['name'](g)}", fontsize=10)
    for m in range(Config.num_views):
        ax[m + 1].imshow((views[m] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        ax[m + 1].set_title(f"view {m + 1}", fontsize=10)
    for a in ax:
        a.axis("off")
    fig.suptitle("Session 1: one of only 5 training images, expanded into deterministic views", fontsize=11)
    fig.tight_layout(); p = os.path.join(OUT, "1_views.png"); fig.savefig(p, dpi=150); plt.close(fig)
    show(p)
    say(f"  Novel class '{S['name'](g)}' gets 5 images; each becomes {Config.num_views} views "
        f"-> {5 * Config.num_views} graph nodes per class.  (opened {os.path.relpath(p, ROOT)})")
    pause("build the session-1 graph and show the learned class topology")

    cls1 = S["sessions"][0]
    t0 = time.time()
    P1, A1, R1 = build_new_class_prototypes_official(S["backbone"], S["stag"], S["train"], refs, cls1,
                                                     S["device"], M=Config.num_views)
    say(f"  Graph over {len(cls1)} classes x 5 shots x {Config.num_views} views built in {time.time() - t0:.2f}s.")
    say("  Learned class-level topology A~ (how strongly the scorer links two classes):")
    nm = [S["name"](c)[:11] for c in cls1]
    say("  " + " " * 13 + "".join(f"{n:>12}" for n in nm))
    A1c = A1.cpu().numpy()
    for i, n in enumerate(nm):
        say(f"  {n:>12} " + "".join(f"{A1c[i, j]:12.2f}" for j in range(len(nm))))
    off = A1c[~np.eye(len(nm), dtype=bool)].mean()
    say(f"  Same class {np.diag(A1c).mean():.2f} vs different classes {off:.2f}: the trained topology barely"
        f" separates classes -- a first hint of Act 3.")
    pause("run all 9 sessions live")

    # live incremental run
    stag, inv_tau = S["stag"], S["stag"].classifier.log_inv_tau.exp()
    H = write_new_memory_rows(stag, C["P"], C["A"], 0, S["device"])
    R, order = C["R"], list(S["base"])
    z = F.normalize(stag.projection(C["f"]), dim=-1)
    fq = F.normalize(C["f"], dim=-1)
    base_set = set(S["base"])
    say(f"\n  {B}{'session':<9}{'classes':>8}   {'NCM  A_B   A_N    HM':>24}   {'STAG-STI residual  A_B   A_N    HM':>36}{RST}")
    res = None
    for s in range(0, len(S["sessions"]) + 1):
        if s >= 1:
            cl = S["sessions"][s - 1]
            Pn, An, Rn = (P1, A1, R1) if s == 1 else build_new_class_prototypes_official(
                S["backbone"], stag, S["train"], official_session_refs(S["spec"], S["plan"], s), cl, S["device"],
                M=Config.num_views)
            H = torch.cat([H, write_new_memory_rows(stag, Pn, An, s, S["device"])])
            R = torch.cat([R, Rn]); order += cl
        res = evaluate(C, z, fq, H, R, order, base_set, inv_tau)
        n, f = res["ncm"], res["fused"]
        tail = "" if s == 0 else f"  {n['A_B']:5.1f} {n['A_N']:5.1f} {n['HM']:5.1f}"
        tail2 = "" if s == 0 else f"  {f['A_B']:5.1f} {f['A_N']:5.1f} {f['HM']:5.1f}"
        say(f"  {s:<9}{len(order):>8}   acc {n['acc']:5.1f}{tail:>18}   acc {f['acc']:5.1f}{tail2:>18}", 0.35)
    gain = res["fused"]["HM"] - res["ncm"]["HM"]
    say(f"\n  After the last session the residual head beats NCM by {GRN}{B}{gain:+.1f} HM{RST}"
        f" ({res['fused']['HM']:.1f} vs {res['ncm']['HM']:.1f}). Looks like the head works...")
    return dict(H=H, R=R, order=order, z=z, fq=fq, inv_tau=inv_tau, base_set=base_set)


def scores(st, lam, b):
    s = st["inv_tau"] * ((1 - lam) * st["s_head"] + lam * st["s_ncm"])
    return s + b * st["novel_col"]


def metrics(pred, st):
    hit = pred == st["y"]
    a_b = hit[st["is_base"]].float().mean().item() * 100
    a_n = hit[~st["is_base"]].float().mean().item() * 100 if (~st["is_base"]).any() else float("nan")
    return dict(acc=hit.float().mean().item() * 100, A_B=a_b, A_N=a_n, HM=hm(a_b, a_n) if a_n == a_n else 0.0)


@torch.no_grad()
def evaluate(C, z, fq, H, R, order, base_set, inv_tau, keep=False):
    pos = {c: i for i, c in enumerate(order)}
    rows = torch.tensor([pos.get(int(t), -1) >= 0 for t in C["y"].tolist()], device=C["y"].device)
    y = torch.tensor([pos[int(t)] for t in C["y"][rows].tolist()], device=C["y"].device)
    st = dict(s_head=z[rows] @ F.normalize(H, dim=-1).T, s_ncm=fq[rows] @ F.normalize(R, dim=-1).T,
              inv_tau=inv_tau, y=y, is_base=torch.tensor([order[i] in base_set for i in y.tolist()], device=y.device),
              novel_col=torch.tensor([c not in base_set for c in order], device=y.device).float(), rows=rows)
    out = dict(ncm=metrics(scores(st, 1.0, B0).argmax(1), st), fused=metrics(scores(st, LAM, B0).argmax(1), st))
    if keep:
        out["st"] = st
    return out


# --------------------------------------------------------------------------
@torch.no_grad()
def act2(S, C, L):
    banner("ACT 2  Real test images from novel classes")
    r = evaluate(C, L["z"], L["fq"], L["H"], L["R"], L["order"], L["base_set"], L["inv_tau"], keep=True)
    st = r["st"]
    p_ncm = scores(st, 1.0, B0).argmax(1)
    p_res = scores(st, LAM, B0).argmax(1)
    p_off = scores(st, 1.0, B0 / LAM).argmax(1)
    test_idx = torch.nonzero(st["rows"]).squeeze(1).tolist()
    # pick novel-class images where NCM is wrong but the residual head is right (the head's "wins")
    cand = [i for i in range(len(test_idx)) if not st["is_base"][i] and p_ncm[i] != st["y"][i] and p_res[i] == st["y"][i]]
    rng = random.Random(7)
    rng.shuffle(cand)
    pick, seen = [], set()
    for i in cand:                                   # one image per class
        if int(st["y"][i]) not in seen:
            pick.append(i); seen.add(int(st["y"][i]))
        if len(pick) == 6:
            break
    order = L["order"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(pick), figsize=(2.6 * len(pick), 3.6))
    say(f"  Test images the residual head gets {GRN}right{RST} and plain NCM gets {RED}wrong{RST}:\n")
    say(f"  {B}{'true class':<16}{'NCM (b=0.5)':<18}{'STAG-STI residual':<20}{'NCM, offset b=1':<18}{RST}")
    for k, i in enumerate(pick):
        nm = lambda p: S["name"](order[int(p)])
        t = nm(st["y"][i])
        mark = lambda p: (GRN if p == st["y"][i] else RED) + f"{nm(p):<18}" + RST
        say(f"  {t:<16}{mark(p_ncm[i])}{mark(p_res[i])}  {mark(p_off[i])}", 0.4)
        pil = S["test"].get_pil(test_idx[i])
        ax[k].imshow(pil.resize((128, 128), resample=0)); ax[k].axis("off")
        ax[k].set_title(f"true: {t}\nNCM: {nm(p_ncm[i])}\nSTAG-STI: {nm(p_res[i])}\nNCM b=1: {nm(p_off[i])}",
                        fontsize=8)
    fig.tight_layout(); p = os.path.join(OUT, "2_predictions.png"); fig.savefig(p, dpi=150); plt.close(fig)
    show(p)
    say(f"\n  {YEL}Notice the last column:{RST} plain NCM with a larger novel-class offset makes the same"
        f" corrections, with no head at all.")
    return dict(st=st, p_res=p_res, p_off=p_off)


@torch.no_grad()
def act3(S, C, L, A2):
    banner("ACT 3  Calibration, not structure")
    st = A2["st"]
    say("  NCM with a constant offset b added to the novel-class scores, swept on the same backbone:\n")
    say(f"  {B}{'b':>5}  {'A_B':>6} {'A_N':>6} {'HM':>6}   HM{RST}")
    best = (0, None)
    for b in [0, 0.5, 1, 1.5, 2, 3, 4, 6]:
        m = metrics(scores(st, 1.0, b).argmax(1), st)
        best = max(best, (m["HM"], b))
        say(f"  {b:>5}  {m['A_B']:6.1f} {m['A_N']:6.1f} {m['HM']:6.1f}   {bar(m['HM'], 40)}", 0.25)
    f = metrics(A2["p_res"], st)
    say(f"  {'head':>5}  {f['A_B']:6.1f} {f['A_N']:6.1f} {f['HM']:6.1f}   {GRN}{bar(f['HM'], 40)}{RST}"
        f"  <- STAG-STI residual")
    pause("check what the head actually does")

    agree = (A2["p_res"] == A2["p_off"]).float().mean().item() * 100
    sh = st["s_head"].std(1).mean().item()
    sn = st["s_ncm"].std(1).mean().item()
    say(f"  Predictions of the residual head vs NCM with offset b = {B0}/{LAM} = {B0 / LAM:g}:"
        f"  {B}{agree:.2f}% identical{RST} over {len(st['y'])} test images")
    say(f"  Spread of class scores per image:  head {sh:.4f}   NCM {sn:.4f}"
        f"   ->  the head is {B}{sn / sh:.0f}x flatter{RST}")
    say(f"\n  The head scores every class almost the same, so (1-{LAM}) x head + {LAM} x NCM is just")
    say(f"  NCM rescaled by 1/{LAM}, which doubles the novel offset. The {f['HM'] - metrics(scores(st, 1.0, B0).argmax(1), st)['HM']:+.1f}"
        f" HM 'gain' is a calibration shift; the best plain offset (b={best[1]:g}) reaches {best[0]:.1f}.")
    say(f"\n  {CYN}{B}Take-away:{RST} compare any FSCIL head or calibration method with NCM over a range of"
        f"\n  offsets on the same backbone. On or below that curve = no recognition gain."
        f"\n  (The paper applies this control to STAG-STI as designed and to TEEN as well.)")


def main():
    global AUTO
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true", help="no pauses (rehearsal / recording)")
    AUTO = ap.parse_args().auto
    t0 = time.time()
    say(f"{DIM}Loading checkpoints and CIFAR-100...{RST}")
    S = setup()
    C = cached(S)
    say(f"{DIM}Ready in {time.time() - t0:.1f}s on {S['device']}.{RST}")
    pause("start")
    L = act1(S, C)
    pause("look at individual predictions")
    A2 = act2(S, C, L)
    pause("explain where the gain comes from")
    act3(S, C, L, A2)
    say("")


if __name__ == "__main__":
    main()
