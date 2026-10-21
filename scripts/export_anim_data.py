"""Export real intermediate tensors of the STAG-STI pipeline (CIFAR-100,
seed 42, held-out residual head) for the pipeline animation page.

    .venv/bin/python scripts/export_anim_data.py
        -> demo/anim_data.json, demo/pipeline_animation.html (+ _local.html, opens offline)

Session 1 (5 novel classes x 5 official shots) is traced through every
stage: fantasy views, backbone features, shot averaging, projection,
learned topology, view-aware GATv2, view pooling, STI memory write. One
real test query is then scored by the head, by NCM, fused, and offset.
"""
import base64
import io
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from fscil.config import Config                                    # noqa: E402
from fscil.data import make_fantasy_views                          # noqa: E402
from fscil.datasets import official_session_refs                   # noqa: E402
from fscil.eval_incremental import build_new_class_prototypes_official, write_new_memory_rows  # noqa: E402
import demo                                                        # noqa: E402


def png(pil, size=64):
    buf = io.BytesIO()
    pil.resize((size, size), resample=0).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def tensor_png(t):
    mean = torch.tensor(Config.data_mean)[:, None, None]
    std = torch.tensor(Config.data_std)[:, None, None]
    arr = ((t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    from PIL import Image
    return png(Image.fromarray(arr))


def r(x, d=4):
    return np.round(np.asarray(x.detach().cpu() if torch.is_tensor(x) else x, dtype=float), d).tolist()


@torch.no_grad()
def main():
    S = demo.setup()
    C = demo.cached(S)
    dev, stag, bb = S["device"], S["stag"], S["backbone"]
    M = Config.num_views
    refs = official_session_refs(S["spec"], S["plan"], 1)
    cls = S["sessions"][0]
    by = {c: [] for c in cls}
    for ref in refs:
        i = S["train"].index_for_ref(ref)
        by[S["train"].targets[i]].append(i)

    # stages 1-3: views -> frozen backbone -> shot-averaged view means
    imgs, views, feats = [], [], []
    for c in cls:
        vc = torch.stack([make_fantasy_views(S["train"].get_pil(i), M) for i in by[c]])  # (5, M, 3, H, W)
        f = bb(vc.view(-1, *vc.shape[-3:]).to(dev)).view(len(by[c]), M, -1)               # (5, M, 512)
        imgs.append([png(S["train"].get_pil(i)) for i in by[c]])
        views.append([[tensor_png(vc[k, m]) for m in range(M)] for k in range(len(by[c]))])
        feats.append(f)
    feats = torch.stack(feats)                       # (5 cls, 5 shots, M, 512)
    w = feats.mean(1)                                # (5, M, 512) shot-averaged
    cid = torch.arange(len(cls), device=dev).repeat_interleave(M)
    vid = torch.arange(M, device=dev).repeat(len(cls))
    h0, h_en, A = stag.encode_prototypes(w.reshape(-1, w.shape[-1]), cid, vid)
    P, A_t = stag.build_prototypes(h_en, A, cid, len(cls))

    # check against the evaluation code path
    P_ref, _, _ = build_new_class_prototypes_official(bb, stag, S["train"], refs, cls, dev, M=M)
    print("max |P - P_eval| =", (P - P_ref).abs().max().item())

    # stage 7: memory (60 base rows + 5 new), stage 8 query scoring at the final session
    H0 = write_new_memory_rows(stag, C["P"], C["A"], 0, dev)
    Hn = write_new_memory_rows(stag, P, A_t, 1, dev)
    # full final-session memory, built exactly as the demo does
    H, R, order = H0, C["R"], list(S["base"])
    for s, cl in enumerate(S["sessions"], start=1):
        Pn, An, Rn = build_new_class_prototypes_official(bb, stag, S["train"],
                                                         official_session_refs(S["spec"], S["plan"], s), cl, dev, M=M)
        H = torch.cat([H, write_new_memory_rows(stag, Pn, An, s, dev)]); R = torch.cat([R, Rn]); order += cl
    z_all = F.normalize(stag.projection(C["f"]), dim=-1)
    fq_all = F.normalize(C["f"], dim=-1)
    inv_tau = stag.classifier.log_inv_tau.exp()
    s_head = z_all @ F.normalize(H, dim=-1).T
    s_ncm = fq_all @ F.normalize(R, dim=-1).T
    novel = torch.tensor([c not in set(S["base"]) for c in order], device=dev).float()
    pos = {c: i for i, c in enumerate(order)}
    y = torch.tensor([pos[int(t)] for t in C["y"].tolist()], device=dev)
    p_ncm = (inv_tau * s_ncm + 0.5 * novel).argmax(1)
    p_fus = (inv_tau * (0.5 * s_head + 0.5 * s_ncm) + 0.5 * novel).argmax(1)
    # a query from a session-1 class that NCM misses and the fused head gets right
    s1 = set(pos[c] for c in cls)
    cand = [i for i in range(len(y)) if int(y[i]) in s1 and p_ncm[i] != y[i] and p_fus[i] == y[i]]
    q = random.Random(3).choice(cand)
    test_idx = q                                     # cache rows follow test-set order
    qpil = S["test"].get_pil(test_idx)

    # 2-D layout: PCA basis fitted on the projected nodes h0 (where classes
    # are separated); refined nodes and prototypes are placed in the same basis
    n0, ne, nP = (F.normalize(t, dim=-1).cpu().numpy() for t in (h0, h_en, P))
    mu = n0.mean(0)
    _, _, Vt = np.linalg.svd(n0 - mu, full_matrices=False)
    proj = lambda X: (X - mu) @ Vt[:2].T
    Yall = np.concatenate([proj(n0), proj(ne), proj(nP)])
    lo, hi = Yall.min(0), Yall.max(0)
    Y = (Yall - lo) / (hi - lo + 1e-9)

    def mean_cos_between(X, labels):
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        Sm = Xn @ Xn.T
        diff = labels[:, None] != labels[None, :]
        return float(Sm[diff].mean())
    lab = cid.cpu().numpy()
    sim = dict(h0_between=mean_cos_between(n0, lab), h_between=mean_cos_between(ne, lab),
               P_between=mean_cos_between(nP, np.arange(len(cls))),
               f_between=mean_cos_between(w.reshape(-1, w.shape[-1]).cpu().numpy(), lab))
    K = h0.shape[0]

    def cos_spread(s):
        return float(s.std().item())

    data = dict(
        classes=[S["name"](c) for c in cls], shots=5, views=M,
        support=imgs, view_imgs=views,
        view_names=["identity", "h-flip", "rotate +15°", "zoom 0.8×"][:M],
        feat_example=r(feats[0, 0, 0], 3), feat_dim=int(feats.shape[-1]),
        w_example=r(w[0, 0], 3), h0_example=r(h0[0], 3), proj_dim=int(h0.shape[-1]),
        A=r(A, 3), A_tilde=r(A_t, 3),
        nodes_h0=r(Y[:K], 3), nodes_h=r(Y[K:2 * K], 3), protos=r(Y[2 * K:], 3),
        node_class=cid.tolist(), node_view=vid.tolist(),
        P_example=r(P[0], 3), H_example=r(Hn[0], 3),
        memory_sizes=[60 + 5 * s for s in range(9)],
        query=dict(img=png(qpil, 96), true=S["name"](order[int(y[q])]), true_pos=int(y[q]),
                   f=r(C["f"][q], 3), z=r(z_all[q], 3),
                   s_head=r(s_head[q], 4), s_ncm=r(s_ncm[q], 4), inv_tau=float(inv_tau),
                   names=[S["name"](c) for c in order], novel=novel.int().tolist(),
                   pred_ncm=S["name"](order[int(p_ncm[q])]), pred_fused=S["name"](order[int(p_fus[q])]),
                   spread_head=cos_spread(s_head[q]), spread_ncm=cos_spread(s_ncm[q])),
        sim=sim,
        feats=r(feats, 2), w_all=r(w.reshape(-1, w.shape[-1]), 2), h0_all=r(h0, 3), h_all=r(h_en, 3),
        P_all=r(P, 3), H_all=r(H, 3), R_all=r(F.normalize(R, dim=-1), 3),
        order_names=[S["name"](c) for c in order], n_base=len(S["base"]),
        agreement=float(((inv_tau * s_ncm + 1.0 * novel).argmax(1) == p_fus).float().mean() * 100),
    )
    out = os.path.join(ROOT, "demo", "anim_data.json")
    json.dump(data, open(out, "w"))
    # build the animation page: demo/pipeline_animation.html (published) and a
    # standalone copy that opens offline by double-click
    html = open(os.path.join(ROOT, "scripts", "pipeline_anim_template.html")).read().replace("/*DATA*/null", json.dumps(data))
    open(os.path.join(ROOT, "demo", "pipeline_animation.html"), "w").write(html)
    open(os.path.join(ROOT, "demo", "pipeline_animation_local.html"), "w").write(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1"></head><body style="margin:0">'
        + html + "</body></html>")
    print("wrote", out, f"{os.path.getsize(out) / 1e6:.2f} MB", "| query:", data["query"]["true"],
          "| NCM:", data["query"]["pred_ncm"], "| fused:", data["query"]["pred_fused"],
          "| spread head/ncm:", round(data["query"]["spread_head"], 4), round(data["query"]["spread_ncm"], 4))


if __name__ == "__main__":
    main()
