import numpy as np
import pandas as pd


STATE = [
    "Electronegativity","Hardness","Electrophilicity","q(N)",
    "f-","f+","f0","s-","s+","s0","s+/s-","s-/s+","s(2)"
]


class RadicalEnv:
    """Reinforcement learning environment for radical optimization."""

    def __init__(self, df: pd.DataFrame, target_omega: float, target_f: dict, bench: list):
        self.df = df.reset_index(drop=True)
        self.sc = STATE
        self.bench = bench
        self.tgt_ω = target_omega
        self.tfm, self.tfp, self.tf0 = target_f["f-"], target_f["f+"], target_f["f0"]

    def reset(self):
        self.idx = np.random.randint(len(self.df))
        return self.df.loc[self.idx, self.sc].values.astype(np.float32)

    def step(self, delta_net, penalty=-5.0):
        s = self.df.loc[self.idx, self.sc].values.astype(np.float32)
        Δω = delta_net(torch.FloatTensor(s).unsqueeze(0)).item()
        s2 = s.copy()
        s2[2] = np.clip(s2[2] + Δω, 0.0, 6.5)
        done = s2[2] >= 5.0
        return s2, -abs(s2[2] - self.tgt_ω) + (penalty if done else 0.0), done, {}

