import os
import pickle
import matplotlib.pyplot as plt


def save_results(results, filepath):
    """Save training results (dict) to pickle."""
    with open(filepath, "wb") as f:
        pickle.dump(results, f)
    print(f"✔ Saved training results → {filepath}")


def load_results(filepath):
    """Load training results (dict) from pickle."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def plot_training_curves(rewards, losses, out_png, title="Training Curves"):
    """Plot reward & loss per episode."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    fig.suptitle(title, fontsize=16)

    axes[0].plot(rewards, linewidth=1.5)
    axes[0].set_title("Reward per Episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True)

    axes[1].plot(losses, linewidth=1.5)
    axes[1].set_title("Loss per Episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_png)
    print(f"✔ Saved training plot → {out_png}")
    plt.close(fig)

