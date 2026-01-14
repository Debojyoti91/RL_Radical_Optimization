from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_ratio(num, den, eps: float = 1e-8) -> np.ndarray:
    """Stable elementwise ratio num/den that avoids division by (near) zero."""
    n = np.asarray(num, dtype=np.float64)
    d = np.asarray(den, dtype=np.float64)
    return n / (np.sign(d) * np.maximum(np.abs(d), eps))


@dataclass
class ZScoreScaler:
    """Simple z-score scaler stored as (mu, sd) Series aligned to columns."""
    mu: pd.Series
    sd: pd.Series

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        sd = self.sd.replace(0.0, 1.0)
        return (df - self.mu) / sd

    def inverse_transform_col(self, x_scaled: np.ndarray, col: str) -> np.ndarray:
        sd = float(self.sd.get(col, 1.0)) or 1.0
        mu = float(self.mu.get(col, 0.0))
        return x_scaled * sd + mu


def preprocess_dataframe(
    df: pd.DataFrame,
    state_columns: List[str],
    omega_col: str,
    fit_scaler: bool = False,
    scaler: Optional[ZScoreScaler] = None,
) -> Tuple[pd.DataFrame, ZScoreScaler]:
    """Prepare a dataframe for RL.

    - Ensures required columns exist
    - Adds missing ratio columns if needed
    - Fills missing values (ffill/bfill)
    - Clips Fukui-like quantities to [-1, 1] and omega to [0, 6.5]
    - Returns z-scored features and a scaler

    Notes
    -----
    The returned df_scaled contains only `state_columns` plus three metadata columns:
      - _omega_unscaled : original omega values (unscaled)
      - _omega_mu, _omega_sd : scalers for omega (for inverse transform)
    """
    df = df.copy()

    # Ensure columns exist
    for c in state_columns:
        if c not in df.columns:
            df[c] = np.nan

    # Add missing ratio columns if expected by state_columns
    if "s+/s-" in state_columns and "s+/s-" not in df.columns:
        df["s+/s-"] = _safe_ratio(df.get("s+", 0.0), df.get("s-", 1.0))
    if "s-/s+" in state_columns and "s-/s+" not in df.columns:
        df["s-/s+"] = _safe_ratio(df.get("s-", 0.0), df.get("s+", 1.0))

    # Fill missing
    df[state_columns] = df[state_columns].ffill().bfill()

    # Clip common CDFT bounds
    for c in ("f-", "f+", "f0"):
        if c in df.columns:
            df[c] = df[c].clip(-1.0, 1.0)

    if omega_col in df.columns:
        df[omega_col] = df[omega_col].clip(0.0, 6.5)

    feats = df[state_columns].copy()

    if fit_scaler or scaler is None:
        mu = feats.mean(numeric_only=True)
        sd = feats.std(numeric_only=True).replace(0.0, 1.0)
        scaler = ZScoreScaler(mu=mu, sd=sd)

    df_scaled = scaler.transform(feats)
    df_scaled["_omega_unscaled"] = df[omega_col].values.astype(np.float32)
    df_scaled["_omega_mu"] = float(scaler.mu.get(omega_col, 0.0))
    df_scaled["_omega_sd"] = float(scaler.sd.get(omega_col, 1.0)) or 1.0

    return df_scaled.reset_index(drop=True), scaler
