from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .envs import ContinuousRadicalEnv
from .models import Actor, Critic
from .replay import ReplayBuffer


def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)


@dataclass
class TrainHistory:
    episode_reward: list
    success: list
    avg_final_dist: list


def train_td3(
    env: ContinuousRadicalEnv,
    *,
    hidden: int = 128,
    lr: float = 1e-4,
    gamma: float = 0.99,
    policy_noise: float = 0.02,
    noise_clip: float = 0.05,
    policy_delay: int = 2,
    num_episodes: int = 500,
    max_steps: int = 50,
    buffer_capacity: int = 200_000,
    batch_size: int = 128,
    warmup: int = 2_000,
    tau: float = 0.005,
    base_success_bonus: float = 10.0,
    late_success_bonus: float = 12.0,
    bonus_boost_start: int = 400,
    anneal_episodes: int = 300,
    pos_clip_start: float = 0.20,
    pos_clip_final: float = 0.10,
    neg_clip_start: float = 0.10,
    neg_clip_final: float = 0.10,
    exploration_sigma_start: float = 0.10,
    exploration_sigma_final: float = 0.02,
    seed: int = 2022,
    device: Optional[str] = None,
) -> Tuple[Actor, Critic, Dict]:
    """Train TD3 on the provided environment.

    Returns
    -------
    actor, critic, history_dict
    """
    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    device_t = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    state_dim = env.num_features

    actor = Actor(state_dim, hidden).to(device_t)
    actor_tgt = Actor(state_dim, hidden).to(device_t)
    actor_tgt.load_state_dict(actor.state_dict())

    critic = Critic(state_dim, hidden).to(device_t)
    critic_tgt = Critic(state_dim, hidden).to(device_t)
    critic_tgt.load_state_dict(critic.state_dict())

    opt_actor = Adam(actor.parameters(), lr=lr)
    opt_critic = Adam(critic.parameters(), lr=lr)

    rb = ReplayBuffer(capacity=buffer_capacity)

    history = TrainHistory(episode_reward=[], success=[], avg_final_dist=[])

    total_steps = 0
    for ep in range(int(num_episodes)):
        # anneal clips and exploration
        frac = min(1.0, ep / max(1.0, float(anneal_episodes)))
        pos_clip = pos_clip_start + (pos_clip_final - pos_clip_start) * frac
        neg_clip = neg_clip_start + (neg_clip_final - neg_clip_start) * frac
        sigma = exploration_sigma_start + (exploration_sigma_final - exploration_sigma_start) * frac

        # ramp success bonus later in training
        env.cfg.success_bonus = late_success_bonus if ep >= bonus_boost_start else base_success_bonus

        s = env.reset()
        ep_r = 0.0
        final_dist = None
        success = 0

        for _ in range(int(max_steps)):
            total_steps += 1

            s_t = torch.from_numpy(s).float().unsqueeze(0).to(device_t)
            with torch.no_grad():
                raw = float(actor(s_t).item())
                a = raw * pos_clip if raw >= 0 else raw * neg_clip
                a = float(a + np.random.normal(0.0, sigma))

            ns, r, done, info = env.step(a, pos_clip=pos_clip, neg_clip=neg_clip)
            ep_r += float(r)
            rb.push(s, a, r, ns, done)
            s = ns

            # distance to target at end of step
            final_dist = abs(float(info["omega"]) - float(env.cfg.target_omega))
            if final_dist <= env.cfg.success_thr:
                success = 1

            # updates
            if len(rb) >= warmup:
                sb, ab, rb_, nsb, db = rb.sample(batch_size)
                sb = torch.from_numpy(sb).float().to(device_t)
                nsb = torch.from_numpy(nsb).float().to(device_t)
                ab = torch.from_numpy(ab).float().to(device_t)
                rb_ = torch.from_numpy(rb_).float().squeeze(1).to(device_t)
                db = torch.from_numpy(db).float().squeeze(1).to(device_t)

                with torch.no_grad():
                    noise = (torch.randn_like(ab) * policy_noise).clamp(-noise_clip, noise_clip)
                    a_tgt = (actor_tgt(nsb).unsqueeze(1) + noise).clamp(-1.0, 1.0)
                    # map to env clip space
                    a_tgt = torch.where(a_tgt >= 0, a_tgt * pos_clip, a_tgt * neg_clip)

                    q1_t, q2_t = critic_tgt(nsb, a_tgt)
                    q_tgt = torch.min(q1_t, q2_t)
                    y = rb_ + (1.0 - db) * gamma * q_tgt

                q1, q2 = critic(sb, ab)
                loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
                opt_critic.zero_grad()
                loss_q.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_critic.step()

                if total_steps % int(policy_delay) == 0:
                    raw_act = actor(sb).unsqueeze(1)
                    a_pi = torch.where(raw_act >= 0, raw_act * pos_clip, raw_act * neg_clip)
                    q1_pi, _ = critic(sb, a_pi)
                    loss_pi = (-q1_pi).mean()
                    opt_actor.zero_grad()
                    loss_pi.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    opt_actor.step()

                    soft_update(actor_tgt, actor, tau)
                    soft_update(critic_tgt, critic, tau)

            if done:
                break

        history.episode_reward.append(ep_r)
        history.success.append(success)
        history.avg_final_dist.append(float(final_dist) if final_dist is not None else float("nan"))

    return actor, critic, {
        "episode_reward": history.episode_reward,
        "success": history.success,
        "avg_final_dist": history.avg_final_dist,
    }
