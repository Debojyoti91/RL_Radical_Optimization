import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
from .models import build_dqn, DeltaNet
from .env import RadicalEnv


def set_seed(seed: int = 42):
    """Ensure reproducibility across random, numpy, torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_agent(env, hidden_size, num_hidden_layers, lr, gamma, eps_decay,
                num_episodes=500, max_steps=10, device="cpu"):
    """Train DQN + DeltaNet agent in RadicalEnv."""
    d_in = len(env.sc)
    dqn = build_dqn(d_in, h=hidden_size, n_layers=num_hidden_layers).to(device)
    delta_model = DeltaNet(d_in).to(device)

    optimizer = optim.Adam(list(dqn.parameters()) + list(delta_model.parameters()), lr=lr)
    criterion = nn.MSELoss()

    rewards, losses = [], []
    eps = 1.0

    for ep in range(num_episodes):
        state = env.reset()
        total_r, loss_sum = 0.0, 0.0

        for step in range(max_steps):
            s_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_vals = dqn(s_tensor)

            if random.random() < eps:
                action = np.random.randint(q_vals.shape[1])
            else:
                action = q_vals.argmax().item()

            next_state, r, done, _ = env.step(delta_model)
            total_r += r

            ns_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_next = dqn(ns_tensor).max().item()
            target = r if done else r + gamma * q_next

            target_vec = q_vals.clone().detach()
            target_vec[0, action] = target
            loss = criterion(q_vals, target_vec)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            state = next_state
            if done:
                break

        eps = max(0.01, eps * eps_decay)
        rewards.append(total_r)
        losses.append(loss_sum / (step + 1))

    return dqn, delta_model, rewards, losses


def grid_search(train_df, val_df, bench, config, results_dir, seed=42, device="cpu"):
    """Perform hyperparameter grid search to find best model."""
    set_seed(seed)

    best_val, best_cfg = -np.inf, None
    gamma = config["gamma"]

    for hs, nh, lr, ed in product(config["hidden_sizes"], config["n_hidden_layers"],
                                  config["lrs"], config["eps_decays"]):
        label = f"hs{hs}_nh{nh}_lr{lr:.0e}_gm{gamma}_ed{ed}"
        print(f"\n=== Running {label} ===")

        fixed_omega, fixed_f = 1.0, {"f-": random.uniform(-1, 1),
                                     "f+": random.uniform(-1, 1),
                                     "f0": random.uniform(-1, 1)}

        env_tr = RadicalEnv(train_df, fixed_omega, fixed_f, bench)
        env_val = RadicalEnv(val_df, fixed_omega, fixed_f, bench)

        dqn, delta_model, rews, losses = train_agent(env_tr, hs, nh, lr, gamma, ed,
                                                     num_episodes=500, max_steps=10,
                                                     device=device)

        # validation
        total_val = 0.0
        for _ in range(100):
            s, ep_r = env_val.reset(), 0.0
            for _ in range(10):
                s_t = torch.FloatTensor(s).unsqueeze(0).to(device)
                with torch.no_grad():
                    a = dqn(s_t).argmax().item()
                s, r, done, _ = env_val.step(delta_model)
                ep_r += r
                if done:
                    break
            total_val += ep_r
        avg_val = total_val / 100

        print(f"{label}: Validation avg reward = {avg_val:.3f}")

        if avg_val > best_val:
            best_val, best_cfg = avg_val, label
            torch.save(dqn.state_dict(), os.path.join(results_dir, "best_model_dqn.pth"))
            torch.save(delta_model.state_dict(), os.path.join(results_dir, "best_model_delta.pth"))
            print(f"✔ Saved new best model: {label} (val={avg_val:.3f})")

    return best_cfg, best_val

