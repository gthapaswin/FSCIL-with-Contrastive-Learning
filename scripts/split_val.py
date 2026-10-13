"""Choose the split-fusion novel lam on the held-out validation (never on test).

Reruns the HeldoutValidator of a finished held-out Phase B run on its best
checkpoint, with base columns at lam_base (default 1 = NCM) and each novel lam,
next to the plain single-lam scores, and writes <run>/split_val.json.

    .venv/bin/python scripts/split_val.py --dataset cub200 --bb reserve_cm30 [--seed 1]
"""
import argparse
import json
import os

import torch

from fscil.backbone import build_backbone
from fscil.config import Config
from fscil.data import get_official_split
from fscil.pipeline import StagStiModel
from fscil.train_session0 import HeldoutValidator, get_device, set_seed, split_heldout

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--bb", required=True, help="backbone tag, e.g. reserve_cm100")
p.add_argument("--seed", type=int, default=None)
p.add_argument("--backbone", default=None, choices=["resnet20"])
p.add_argument("--lam_base", type=float, default=1.0)
p.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
args = p.parse_args()

spec = Config.apply_dataset(args.dataset)
if args.backbone == "resnet20":
    Config.backbone_type, Config.backbone_out_dim = "cifar_resnet20", 64
sub = "" if args.seed in (None, Config.seed) else f"seed{args.seed}"
if args.seed is not None:
    Config.seed = args.seed
bb_dir = os.path.join(Config.ckpt_dir, args.bb, sub)
run_dir = os.path.join(Config.ckpt_dir, args.bb + "_heldout", sub)
set_seed(Config.seed)
device = get_device(Config.device)

backbone = build_backbone(Config.backbone_type, out_dim=Config.backbone_out_dim).to(device)
backbone.load_state_dict(torch.load(os.path.join(bb_dir, "backbone_base.pt"), map_location=device))
backbone.freeze()
stag = StagStiModel().to(device)
stag.load_state_dict(torch.load(os.path.join(run_dir, "stag_sti_session0_best.pt"), map_location=device))

base_classes, _ = get_official_split(spec)
train_classes, held_classes = split_heldout(base_classes)
assert held_classes == json.load(open(os.path.join(run_dir, "heldout_val.json")))["held_classes"]
validator = HeldoutValidator(backbone, train_classes, held_classes, device)

single = validator(stag, args.lams)
split = validator(stag, [(args.lam_base, l) for l in args.lams])
out = {"lam_base": args.lam_base,
       "single": {str(k): v for k, v in single.items()},
       "split": {str(k[1]): v for k, v in split.items()}}
out["selected_split_lam"] = float(max(split, key=lambda k: split[k]["hm"])[1])
for name, d in [("single", single), ("split", split)]:
    for k, v in d.items():
        print(f"{name:6s} lam={k}: seen {v['seen']*100:.2f} held {v['held']*100:.2f} HM {v['hm']*100:.2f}")
print("selected split novel lam:", out["selected_split_lam"])
json.dump(out, open(os.path.join(run_dir, "split_val.json"), "w"), indent=2)
