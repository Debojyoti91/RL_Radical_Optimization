from __future__ import annotations

import collections
import random
from typing import Deque, Tuple

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.buf: Deque[Tuple[np.ndarray, float, float, np.ndarray, bool]] = collections.deque(
            maxlen=int(capacity)
        )

    def push(self, s: np.ndarray, a: float, r: float, ns: np.ndarray, d: bool) -> None:
        self.buf.append((s.astype(np.float32), float(a), float(r), ns.astype(np.float32), bool(d)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, int(batch_size))
        s, a, r, ns, d = map(np.array, zip(*batch))
        return (
            s,
            a.reshape(-1, 1).astype(np.float32),
            r.reshape(-1, 1).astype(np.float32),
            ns,
            d.reshape(-1, 1).astype(np.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)
