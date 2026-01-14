from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class EnvConfig:
    state_columns: List[str]
    omega_col_index: int = 2  # index of ω within the state vector
    target_omega: float = 1.0
    success_thr: float = 0.05
    rmax: float = 2.0
    success_bonus: float = 10.0
    omega_clip: Tuple[float, float] = (0.0, 6.5)
    high_omega_terminate: float = 5.0
    high_omega_penalty: float = -5.0
    start_policy: str = "mixed"  # 'below', 'above', 'mixed'


class ContinuousRadicalEnv:
    """1D action environment that edits ω only (scaled in state vector).

    State: vector of z-scored CDFT descriptors.
    Action: scalar delta applied to ω (in unscaled space), then re-scaled into state.

    Reward:
      proximity(new_omega) + 0.2 * (prev_dist - new_dist) + success_bonus(if within thr) + penalty
    """
    def __init__(
        self,
        df_scaled: pd.DataFrame,
        cfg: EnvConfig,
    ):
        self.df = df_scaled.reset_index(drop=True)
        self.cfg = cfg
        self.state_columns = cfg.state_columns
        self.num_features = len(self.state_columns)

        self.state: np.ndarray | None = None
        self.idx: int | None = None
        self.row_cache: pd.Series | None = None

    def set_success_threshold(self, thr: float) -> None:
        self.cfg.success_thr = float(thr)

    def set_start_policy(self, policy: str) -> None:
        self.cfg.start_policy = str(policy)

    def reset(self) -> np.ndarray:
        df = self.df
        if self.cfg.start_policy in ("below", "above"):
            if self.cfg.start_policy == "below":
                mask = df["_omega_unscaled"] < self.cfg.target_omega
            else:
                mask = df["_omega_unscaled"] > self.cfg.target_omega
            sub = df[mask]
            if len(sub) > 0:
                df = sub

        self.idx = int(np.random.choice(df.index))
        self.state = df.loc[self.idx, self.state_columns].values.astype(np.float32)
        self.row_cache = df.loc[self.idx]
        return self.state.copy()

    def reset_idx(self, idx: int) -> np.ndarray:
        idx = int(idx)
        self.idx = idx
        self.state = self.df.loc[idx, self.state_columns].values.astype(np.float32)
        self.row_cache = self.df.loc[idx]
        return self.state.copy()

    def _success_bonus(self, omega: float) -> float:
        return self.cfg.success_bonus if abs(omega - self.cfg.target_omega) <= self.cfg.success_thr else 0.0

    def _proximity_reward(self, omega: float) -> float:
        dist = abs(omega - self.cfg.target_omega)
        return max(0.0, 1.0 - (dist / self.cfg.rmax) ** 2)

    def step(self, delta_omega: float, pos_clip: float, neg_clip: float):
        if self.state is None or self.row_cache is None:
            raise RuntimeError("Environment must be reset before calling step().")

        omega_mu = float(self.row_cache["_omega_mu"])
        omega_sd = float(self.row_cache["_omega_sd"]) or 1.0

        w_idx = int(self.cfg.omega_col_index)
        omega_scaled = float(self.state[w_idx])
        prev_omega = omega_sd * omega_scaled + omega_mu
        prev_dist = abs(prev_omega - self.cfg.target_omega)

        a = float(np.clip(delta_omega, -float(neg_clip), +float(pos_clip)))
        new_omega = float(np.clip(prev_omega + a, *self.cfg.omega_clip))
        new_dist = abs(new_omega - self.cfg.target_omega)

        done, penalty = False, 0.0
        if new_omega >= self.cfg.high_omega_terminate:
            done, penalty = True, float(self.cfg.high_omega_penalty)
        if new_dist <= self.cfg.success_thr:
            done = True

        proximity = self._proximity_reward(new_omega)
        progress = 0.2 * (prev_dist - new_dist)
        bonus = self._success_bonus(new_omega)
        reward = proximity + progress + bonus + penalty

        new_omega_scaled = (new_omega - omega_mu) / omega_sd
        new_state = self.state.copy()
        new_state[w_idx] = float(new_omega_scaled)
        self.state = new_state

        info: Dict[str, float] = {"omega": float(new_omega), "delta_clamped": float(a)}
        return new_state.copy(), float(reward), bool(done), info
