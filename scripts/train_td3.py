#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from rlcdft.preprocess import preprocess_dataframe
from rlcdft.envs import EnvConfig, ContinuousRadicalEnv
from rlcdft.td3 import train_td3
from rlcdft.plotting import plot_training_curves, plot_episode_trace


DEFAULT_STATE_COLS = [
    "Electronegativity","Hardness","Electrophilicity","q(N)",
    "f-","f+","f0","s-","s+","s0","s+/s-","s-/s+","s(2)"
]


def parse_args():
    p = argparse.ArgumentParser(description="Train TD3 to tune omega (ω) toward a target using CDFT descriptors.")
    p.add_argument("--data", type=str, required=True, help="CSV file containing CDFT descriptor columns.")
    p.add_argument("--outdir", type=str, default="outputs", help="Output directory.")
    p.add_argument("--omega-col", type=str, default="Electrophilicity", help="Column name for ω.")
    p.add_argument("--state-cols", type=str, default=",".join(DEFAULT_STATE_COLS),
                   help="Comma-separated list of state columns (must include omega-col).")
    p.add_argument("--target-omega", type=float, default=1.0)
    p.add_argument("--success-thr", type=float, default=0.05)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--device", type=str, default=None, help="cpu, cuda, or cuda:0 ...")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    state_cols = [c.strip() for c in args.state_cols.split(",") if c.strip()]
    if args.omega_col not in state_cols:
        raise SystemExit(f"--omega-col '{args.omega_col}' must be included in --state-cols")

    df_raw = pd.read_csv(args.data)
    df_scaled, scaler = preprocess_dataframe(df_raw, state_columns=state_cols, omega_col=args.omega_col, fit_scaler=True)

    cfg = EnvConfig(
        state_columns=state_cols,
        omega_col_index=state_cols.index(args.omega_col),
        target_omega=float(args.target_omega),
        success_thr=float(args.success_thr),
    )
    env = ContinuousRadicalEnv(df_scaled, cfg)

    actor, critic, history = train_td3(
        env,
        hidden=int(args.hidden),
        lr=float(args.lr),
        num_episodes=int(args.episodes),
        max_steps=int(args.steps),
        seed=int(args.seed),
        device=args.device,
    )

    # Save artifacts
    torch.save(actor.state_dict(), outdir / "actor_td3.pth")
    torch.save(critic.state_dict(), outdir / "critic_td3.pth")
    (outdir / "history.json").write_text(json.dumps(history, indent=2))

    # Plots
    plot_training_curves(history, outdir / "train_curves")
    plot_episode_trace(env, actor, steps=min(50, args.steps), outpath=outdir / "episode_trace.png", device=args.device)

    # Config snapshot
    (outdir / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    print(f"[OK] Saved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
