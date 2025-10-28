from dataclasses import dataclass
from typing import final

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch import Tensor
from torch._prims_common import DeviceLikeType


@final
class ReplayBuffer:
    """Replay buffer for storing and sampling interactions"""

    def __init__(self, capacity: int, state_space: Box, action_space: Discrete):
        self.capacity = capacity
        self.state_space = state_space
        self.action_space = action_space

        # buffers
        self.next_index = 0
        self.num_inserts = 0
        self.states = np.zeros((capacity, *state_space.shape), dtype=state_space.dtype)
        self.next_states = np.zeros(
            (capacity, *state_space.shape), dtype=state_space.dtype
        )
        self.actions = np.zeros((capacity, 1), dtype=action_space.dtype)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.terminateds = np.zeros((capacity,), dtype=bool)

    def push(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
    ):
        """Inserts a new interaction into the replay buffer"""
        i = self.next_index
        self.states[i] = state
        self.next_states[i] = next_state
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminateds[i] = terminated
        self.next_index = (i + 1) % self.capacity
        self.num_inserts += 1

    def sample(self, batch_size: int) -> "ReplayBatch":
        """Returns a batch of random samples from the replay buffer"""
        indices = np.random.randint(0, len(self), size=batch_size)
        return ReplayBatch(
            states=self.states[indices],
            next_states=self.next_states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            terminateds=self.terminateds[indices],
        )

    def __len__(self) -> int:
        return min(self.capacity, self.num_inserts)


@dataclass
class ReplayBatch:
    states: np.ndarray
    next_states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminateds: np.ndarray

    def to_torch(self, device: DeviceLikeType) -> "TorchReplayBatch":
        """Converts to PyTorch tensors"""
        return TorchReplayBatch(
            states=torch.as_tensor(self.states, device=device),
            next_states=torch.as_tensor(self.next_states, device=device),
            actions=torch.as_tensor(self.actions, device=device),
            rewards=torch.as_tensor(self.rewards, device=device),
            terminateds=torch.as_tensor(self.terminateds, device=device),
        )


@dataclass
class TorchReplayBatch:
    states: Tensor
    next_states: Tensor
    actions: Tensor
    rewards: Tensor
    terminateds: Tensor
