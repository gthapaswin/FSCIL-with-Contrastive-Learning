"""
Central configuration for STAG-STI (Spatio-Temporal Graph Integration and
Contrastive Pre-Conditioning for FSCIL) — Session 0 (base) training on CIFAR-100.

Edit values here rather than hunting through the training script.
"""

import os

class Config:
    # ---------------- Paths ----------------
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

    # ---------------- Dataset / base-novel split ----------------
    # Fixed-seed random 60/40 split (60 base classes; the other 40 are
    # shuffled and cut into 8 groups of 5 for the incremental sessions).
    num_base_classes = 60
    num_novel_classes = 40
    num_incremental_sessions = 8
    way = 5
    shot = 5

    image_size = 32          # CIFAR-100 native resolution, no resize needed
    cifar_mean = (0.5071, 0.4865, 0.4409)
    cifar_std = (0.2673, 0.2564, 0.2762)

    # ---------------- Backbone ----------------
    backbone_out_dim = 512   # d
    backbone_pretrain_epochs = 30
    backbone_pretrain_lr = 0.1
    backbone_pretrain_batch_size = 128
    backbone_pretrain_weight_decay = 1e-3     # was 5e-4 -- increased to fight the 99%/76% train/val gap
    label_smoothing = 0.1                        # softens both Phase A and Phase B cross-entropy targets
    random_erasing_prob = 0.25                    # cutout-style augmentation added to train transforms
    early_stop_patience = 8                          # epochs of no val_acc improvement before Phase A stops early

    # ---------------- Graph / projection dims ----------------
    proj_dim = 256           # d' (post W_proj bottleneck)
    supcon_proj_dim = 128    # dimensionality of the SupCon hypersphere head output
    gat_hidden_dim = 256     # d_attn
    view_rel_dim = 4         # d_r, dimensionality of the view-transform relation embedding
    dropout_p = 0.1            # dropout added inside GATv2 / topology scorer to fight Phase B overfitting

    # ---------------- Multi-view fantasy augmentation ----------------
    num_views = 4             # M

    # ---------------- Episodic training (Phase B: main_epochs) ----------------
    main_epochs = 50          # <-- the "50 epochs" requested by your guide
    episodes_per_epoch = 100
    episode_way = 15          # C_t per episode during base-session episodic training
    episode_shot = 5          # S
    episode_query = 5         # queries per class per episode

    # ---------------- Loss weights (per your Session-0 composite loss table) ----------------
    supcon_temperature = 0.07
    lambda_cls = 1.0            # L_cls (cross-entropy on query classification)
    lambda_supcon = 0.5         # lambda_1 * L_supcon_view
    lambda_graph = 0.25         # lambda_2 * L_graph (topology BCE supervision)
    lambda_stab = 0.1           # lambda_3 * L_stab (memory adapter + gate regularizer)
    lambda_gate_sat = 0.1       # weight of L_gate_sat *inside* L_stab (separate from lambda_stab)
    graph_loss_eps = 1e-7        # BCE numerical-stability clamp

    # ---------------- STI memory: Age-Based Stability Prior ----------------
    kappa = 2.0                  # slope/growth rate -> hardening speed
    kappa0 = 3.0                  # offset bias -> new classes start near-unfrozen (beta~0)

    # ---------------- Optimizer (Phase B) ----------------
    main_lr = 1e-3
    weight_decay = 1e-2          # was 5e-4 -- increased to fight the 99%/84% train/val gap
    main_early_stop_patience = 12  # epochs of no val_query_acc improvement before Phase B stops early

    # ---------------- Misc ----------------
    seed = 42
    num_workers = 2            # keep low on Mac laptops
    device = "auto"             # "auto" picks mps -> cuda -> cpu
    log_every = 10
