"""
Central configuration for STAG-STI (Spatio-Temporal Graph Integration and
Contrastive Pre-Conditioning for FSCIL) — Session 0 (base) training on CIFAR-100.

Edit values here rather than hunting through the training script.
"""

import os

class Config:
    # ---------------- Paths ----------------
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(repo_root, "data")
    # ckpt_dir / log_dir are namespaced per dataset by apply_dataset(); the
    # values below are the cifar100 defaults so importing Config without an
    # explicit apply_dataset() call still behaves like the original pipeline.
    ckpt_dir = os.path.join(repo_root, "checkpoints", "cifar100")
    log_dir = os.path.join(repo_root, "logs", "cifar100")

    # ---------------- Dataset / base-novel split ----------------
    # Which benchmark is active. Set via Config.apply_dataset("miniimagenet")
    # etc. -- see fscil/datasets.py for the registry. The official CEC/FACT
    # class splits (vendored under fscil/splits/) are used for reproducibility.
    dataset = "cifar100"
    num_base_classes = 60
    num_novel_classes = 40
    num_incremental_sessions = 8
    way = 5
    shot = 5

    image_size = 32          # CIFAR-100 native resolution, no resize needed
    backbone_type = "cifar_resnet18"   # cifar_resnet18 | resnet18 | resnet18_pretrained
    # Active normalization stats. data_mean/data_std are the canonical names;
    # cifar_mean/cifar_std are kept as aliases for backward compatibility.
    data_mean = (0.5071, 0.4865, 0.4409)
    data_std = (0.2673, 0.2564, 0.2762)
    cifar_mean = data_mean
    cifar_std = data_std

    # ---------------- Backbone ----------------
    backbone_out_dim = 512   # d
    backbone_pretrain_epochs = 30
    backbone_pretrain_lr = 0.1
    backbone_pretrain_batch_size = 128
    backbone_pretrain_weight_decay = 1e-3     # was 5e-4 -- increased to fight the 99%/76% train/val gap
    label_smoothing = 0.1                        # softens both Phase A and Phase B cross-entropy targets
    random_erasing_prob = 0.25                    # cutout-style augmentation added to train transforms
    early_stop_patience = 8                          # epochs of no val_acc improvement before Phase A stops early
    use_mixup_cutmix = True                            # Phase A: randomly apply Mixup or CutMix per batch
    mixup_alpha = 0.2                                     # Beta(alpha,alpha) mixing coefficient for Mixup
    cutmix_alpha = 1.0                                      # Beta(alpha,alpha) mixing coefficient for CutMix

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
    supcon_temperature = 0.15   # was 0.07 -- raised to loosen over-tight clustering (val_supcon was flat/noisy)
    lambda_cls = 1.0            # L_cls (cross-entropy on query classification)
    lambda_supcon = 0.3         # was 0.5 -- reduced so SupCon doesn't over-constrain the novel-class feature space
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

    # ---------------- Incremental eval-time calibration (Phase C) ----------------
    # ASSUMPTION / heuristic: not derived from any spec -- base prototypes are
    # averaged from many more images and trained on longer than 5-shot novel
    # prototypes, so even after L2-normalizing both (cosine similarity is
    # already magnitude-invariant), base-class cosine scores tend to run
    # systematically higher just from being better-estimated points. This is
    # a simple additive logit bias applied ONLY to non-base (novel) classes
    # at incremental-eval time to compensate. Tune this against your own
    # val split -- 0.0 disables it entirely.
    novel_logit_bias = 0.5

    # ---------------- Misc ----------------
    seed = 42
    num_workers = 2            # keep low on Mac laptops
    device = "auto"             # "auto" picks mps -> cuda -> cpu
    log_every = 10

    # -----------------------------------------------------------------------
    # Dataset switch
    # -----------------------------------------------------------------------
    @classmethod
    def apply_dataset(cls, dataset_key):
        """Mutate the dataset-specific fields to match one benchmark from the
        registry (fscil/datasets.py). All downstream code reads Config.*, so
        this one call reconfigures the whole pipeline. Output dirs are
        namespaced per dataset so runs never clobber each other."""
        from fscil.datasets import get_spec
        spec = get_spec(dataset_key)
        cls.dataset = spec.key
        cls.spec = spec
        cls.image_size = spec.image_size
        cls.backbone_type = spec.backbone
        cls.data_mean = spec.mean
        cls.data_std = spec.std
        cls.cifar_mean = spec.mean   # backward-compat aliases
        cls.cifar_std = spec.std
        cls.num_base_classes = spec.num_base_classes
        cls.num_novel_classes = spec.num_novel_classes
        cls.num_incremental_sessions = spec.num_incremental_sessions
        cls.way = spec.way
        cls.shot = spec.shot
        cls.ckpt_dir = os.path.join(cls.repo_root, "checkpoints", spec.key)
        cls.log_dir = os.path.join(cls.repo_root, "logs", spec.key)
        # keep episodic way within the available base-class pool
        cls.episode_way = min(cls.episode_way, spec.num_base_classes)
        return spec
