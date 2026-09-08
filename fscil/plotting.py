"""
Generates comparison plots from the training/eval artifacts saved by
train_session0.py and eval_incremental.py:
  - checkpoints/plots/phase_a_curves.png       (backbone: loss + acc, train vs val)
  - checkpoints/plots/phase_b_curves.png       (STAG-STI: total loss + acc, train vs val)
  - checkpoints/plots/phase_b_loss_components.png  (cls/supcon/graph/stab breakdown)
  - checkpoints/plots/incremental_accuracy.png (session-by-session accuracy curve)

Run after training and/or after the incremental eval:
    python -m fscil.plotting
"""

import json
import os

import matplotlib
matplotlib.use("Agg")   # no display needed, just save PNGs
import matplotlib.pyplot as plt

from fscil.config import Config


def _save(fig, name, out_dir):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_phase_a(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Phase A (backbone pretrain): loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], label="train_acc")
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], label="val_acc")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy (%)")
    axes[1].set_title("Phase A (backbone pretrain): accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "phase_a_curves.png", out_dir)


def plot_phase_b(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="train_loss (total)")
    axes[0].plot(epochs, history["val_loss"], label="val_loss (total)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("composite loss")
    axes[0].set_title("Phase B (STAG-STI): total loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], label="train_query_acc")
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], label="val_query_acc")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("query accuracy (%)")
    axes[1].set_title("Phase B (STAG-STI): episodic query accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "phase_b_curves.png", out_dir)

    # loss component breakdown, train vs val, one subplot per component
    fig2, axes2 = plt.subplots(2, 2, figsize=(11, 8))
    components = ["cls", "supcon", "graph", "stab"]
    for ax, comp in zip(axes2.flat, components):
        ax.plot(epochs, history[f"train_{comp}"], label=f"train_{comp}")
        ax.plot(epochs, history[f"val_{comp}"], label=f"val_{comp}")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.set_title(f"Phase B: {comp} loss component"); ax.legend(); ax.grid(alpha=0.3)
    fig2.tight_layout()
    _save(fig2, "phase_b_loss_components.png", out_dir)


def plot_incremental(results_path, out_dir):
    with open(results_path) as f:
        data = json.load(f)
    results = data["results"]
    sessions = [r["session"] for r in results]
    accs = [r["accuracy"] * 100 for r in results]
    num_classes = [r["num_classes_seen"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sessions, accs, marker="o", color="tab:blue")
    for s, a, n in zip(sessions, accs, num_classes):
        ax.annotate(f"{a:.1f}%\n({n} cls)", (s, a), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_xlabel("session (0 = base)")
    ax.set_ylabel("cumulative test accuracy (%)")
    ax.set_title(f"Incremental FSCIL accuracy per session  |  PD = {data['PD']:.2f} pts, "
                 f"avg = {data['avg_accuracy']:.2f}%")
    ax.set_xticks(sessions)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "incremental_accuracy.png", out_dir)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="cifar100",
                    help="cifar100 | miniimagenet | cub200")
    ap.add_argument("--ablate", nargs="+", default=None)
    args = ap.parse_args()
    Config.apply_dataset(args.dataset)
    Config.apply_ablation(args.ablate)

    out_dir = os.path.join(Config.ckpt_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    history_path = os.path.join(Config.ckpt_dir, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            full_history = json.load(f)
        if "phase_a" in full_history:
            plot_phase_a(full_history["phase_a"], out_dir)
        else:
            print("No Phase A history found in training_history.json (was --skip_backbone_pretrain used?)")
        if "phase_b" in full_history:
            plot_phase_b(full_history["phase_b"], out_dir)
    else:
        print(f"No training history found at {history_path} -- run train_session0.py first.")

    results_path = os.path.join(Config.ckpt_dir, "incremental_results.json")
    if os.path.exists(results_path):
        plot_incremental(results_path, out_dir)
    else:
        print(f"No incremental results found at {results_path} -- run eval_incremental.py first.")


if __name__ == "__main__":
    main()
