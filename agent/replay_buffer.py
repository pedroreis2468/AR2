"""
Replay Buffer para SAC.
Armazena transições (s, a, r, s', done) para treino off-policy.
"""
import numpy as np
from typing import Tuple


class ReplayBuffer:
    """Replay buffer circular com amostragem uniforme."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float,
            next_obs: np.ndarray, done: bool):
        """Adiciona uma transição ao buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Amostra um batch aleatório do buffer."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.observations[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_observations[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self.size
