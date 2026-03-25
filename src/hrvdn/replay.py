from __future__ import annotations

from collections import deque
from typing import Any, Dict, List

import numpy as np


class SequenceReplayBuffer:
    def __init__(self, capacity: int = 900):
        self.capacity = capacity
        self.data = deque(maxlen=capacity)

    def push(self, transition: Dict[str, Any]):
        self.data.append(transition)

    def __len__(self):
        return len(self.data)

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.data), size=batch_size, replace=False)
        return [self.data[i] for i in idx]

    def recalc_rewards(self, fn):
        for item in self.data:
            item["reward"] = fn(item["reward_ctx"])

