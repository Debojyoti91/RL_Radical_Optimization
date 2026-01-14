from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from .envs import ContinuousRadicalEnv
from .models import Actor


def plot_training_curves(history: Dict, outpath: str | Path) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    rewards = np.asarray(history.get("episode_reward", []), dtype=float)
    success = np.asarray(history.get("success", []), dtype=float)

    if rewards.size:
        plt.figure()
        plt.plot(rewards)
        plt.grid(True)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("TD3 training: episode return")
        plt.tight_layout()
        plt.savefig(outpath.with_suffix(".rewards.png"), dpi=300)
        plt.close()

    if success.size:
        plt.figure()
        # running mean (window 20)
        w = 20
        if len(success) >= w:
            rm = np.convolve(success, np.ones(w) / w, mode="valid")
            plt.plot(np.arange(w - 1, w - 1 + len(rm)), rm)
        else:
            plt.plot(success)
        plt.grid(True)
        plt.xlabel("Episode")
        plt.ylabel("Success rate")
        plt.title("TD3 training: success rate (running mean)")
        plt.tight_layout()
        plt.savefig(outpath.with_suffix(".success.png"), dpi=300)
        plt.close()


def plot_episode_trace(
    env: ContinuousRadicalEnv,
    actor: Actor,
    *,
    steps: int = 50,
    pos_clip: float = 0.10,
    neg_clip: float = 0.10,
    outpath: str | Path = "episode_trace.png",
    device: Optional[str] = None,
) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    s = env.reset()
    omegas = []

    for _ in range(int(steps)):
        s_t = torch.from_numpy(s).float().unsqueeze(0).to(device_t)
        with torch.no_grad():
            raw = float(actor(s_t).item())
        a = raw * pos_clip if raw >= 0 else raw * neg_clip
        ns, r, done, info = env.step(a, pos_clip=pos_clip, neg_clip=neg_clip)
        omegas.append(float(info["omega"]))
        s = ns
        if done:
            break

    plt.figure()
    plt.plot(omegas, marker="o", linewidth=1)
    plt.grid(True)
    plt.xlabel("Step")
    plt.ylabel("ω (unscaled)")
    plt.title("Episode trace: ω")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
