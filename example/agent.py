import typing
from dataclasses import dataclass
from typing import final

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.optim import AdamW

import eval_env


@final
class CartPoleAgent:
    def __init__(
        self,
        seed: int = 1000,
        steps_per_epoch: int = 4000,
        epochs: int = 120,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        pi_lr: float = 1e-3,
        value_lr: float = 1e-2,
        train_policy_iters: int = 80,
        train_value_iters: int = 80,
        lamda: float = 0.97,
        max_episode_length: int = 1000,
        target_kl: float = 0.05,
    ):
        super().__init__()

        env_name = "CartPoleCustom-v0"

        self.env: gym.Env[np.ndarray, int] = gym.make(env_name)  # pyright: ignore[reportUnknownMemberType]
        state_space: Box = self.env.observation_space  # pyright: ignore[reportAssignmentType]
        action_space: Discrete = self.env.action_space  # pyright: ignore[reportAssignmentType]

        # RNG seeds
        np.random.seed(seed)
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
        self.env.action_space.seed(seed)  # pyright: ignore[reportUnusedCallResult]
        self.seed = seed

        # trajectory buffer
        self.trajectories = TrajectoryBuffer(
            capacity=steps_per_epoch,
            state_shape=list(state_space.shape),
            action_shape=[1],
            gamma=gamma,
            lamda=lamda,
        )

        # hyperparameters
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.train_policy_iters = train_policy_iters
        self.train_value_iters = train_value_iters
        self.lamda = lamda
        self.max_episode_length = max_episode_length
        self.target_kl = target_kl

        # models
        self.actor_critic = ActorCritic(
            state_space=state_space,
            action_space=action_space,
        )
        self.policy_optimizer = AdamW(self.actor_critic.pi.parameters(), lr=pi_lr)
        self.value_optimizer = AdamW(self.actor_critic.v.parameters(), lr=value_lr)

    def train(self):
        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}")
            state, _ = self.env.reset()

            for t in range(self.steps_per_epoch):
                action, logp_action, value = self.actor_critic.step(
                    torch.as_tensor(state, dtype=torch.float32)
                )
                next_state, reward, done, truncated, _ = self.env.step(action)

                self.trajectories.push(
                    state=state,
                    action=action,
                    logp=float(logp_action),
                    value=float(value),
                    reward=float(reward),
                )

                state = next_state
                if done or truncated or (t == self.steps_per_epoch - 1):
                    _, _, value = self.actor_critic.step(
                        torch.as_tensor(state, dtype=torch.float32)
                    )
                    is_truncated = truncated or t == self.steps_per_epoch - 1
                    self.trajectories.push_episode_end(value, is_truncated=is_truncated)
                    state, _ = self.env.reset()

            self.update()

    def update(self):
        batch = self.trajectories.get_batch().as_torch()

        policy_loss_old, _ = self.policy_loss(batch)
        value_loss_old = self.value_loss(batch)

        # train policy net
        for i in range(self.train_policy_iters):
            self.policy_optimizer.zero_grad()
            policy_loss, policy_info = self.policy_loss(batch)
            if policy_info.approximate_kl > 1.5 * self.target_kl:
                print(f"early stopping at step {i} due to reaching max KL")
                break
            policy_loss.backward()  # pyright: ignore[reportUnknownMemberType]
            self.policy_optimizer.step()  # pyright: ignore[reportUnknownMemberType]

        # train value net
        for i in range(self.train_value_iters):
            self.value_optimizer.zero_grad()
            value_loss = self.value_loss(batch)
            value_loss.backward()  # pyright: ignore[reportUnknownMemberType]
            self.value_optimizer.step()  # pyright: ignore[reportUnknownMemberType]

        print("Losses:")
        print(f"Policy Loss: {policy_loss_old.item():.4f} → {policy_loss.item():.4f}")
        print(f"Value Loss: {value_loss_old.item():.4f} → {value_loss.item():.4f}")

    def policy_loss(self, batch: "TrajectoryTorchBatch") -> tuple[Tensor, "PolicyInfo"]:
        # policy loss
        pi: Categorical
        logps: Tensor
        pi, logps = self.actor_critic.pi(batch.states, batch.actions)  # pyright: ignore[reportAny]
        ratio = torch.exp(logps - batch.logps)
        min = 1 - self.clip_ratio
        max = 1 + self.clip_ratio
        clipped_adv = torch.clamp(ratio, min, max) * batch.advantages
        unclipped = ratio * batch.advantages
        policy_loss = -torch.min(unclipped, clipped_adv).mean()

        # additional policy info
        approximate_kl = (batch.logps - logps).mean().item()
        mean_entropy = pi.entropy().mean().item()
        clipped_fraction = (
            (ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio))
            .to(torch.float32)
            .mean()
            .item()
        )
        policy_info = PolicyInfo(
            approximate_kl=approximate_kl,
            mean_entropy=mean_entropy,
            clipped_fraction=clipped_fraction,
        )

        return policy_loss, policy_info

    def value_loss(self, batch: "TrajectoryTorchBatch") -> Tensor:
        return ((self.actor_critic.v(batch.states) - batch.returns) ** 2).mean()


@final
class CategoricalActor(nn.Module):
    """Policy prediction over discrete 1D action spaces"""

    def __init__(
        self,
        d_state: int,
        d_hidden: int,
        d_action: int,
        activation: nn.Module,
    ):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.logits_net = nn.Sequential(
            nn.Linear(d_state, d_hidden),
            activation(),
            nn.Linear(d_hidden, d_hidden),
            activation(),
            nn.Linear(d_hidden, d_action),
        )

    def forward(
        self,
        states: Tensor,
        actions: Tensor | None = None,
    ) -> tuple[Categorical, Tensor | None]:
        policy = self.policy(states)
        logp = None
        if actions is not None:
            logp = self.logprob(policy, actions)
        return policy, logp

    def policy(self, states: Tensor) -> Categorical:
        logits = typing.cast(Tensor, self.logits_net(states))  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
        return Categorical(logits=logits)

    def logprob(self, policy: Categorical, action: Tensor) -> Tensor:
        action = action.squeeze(-1) if action.dim() > 1 else action
        return typing.cast(Tensor, policy.log_prob(action))


@final
class Critic(nn.Module):
    """Value prediction over 1D spaces"""

    def __init__(self, d_state: int, d_hidden: int, activation: nn.Module):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.value_net = nn.Sequential(
            nn.Linear(d_state, d_hidden),
            activation(),
            nn.Linear(d_hidden, d_hidden),
            activation(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, states: Tensor) -> Tensor:
        return self.value_net(states).squeeze(-1)


@final
class ActorCritic(nn.Module):
    def __init__(
        self,
        state_space: Box,
        action_space: Box | Discrete,
        d_hidden: int = 3,
        activation: nn.Module = nn.Tanh,  # pyright: ignore[reportArgumentType]
    ):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        if isinstance(action_space, Box):
            self.pi = CategoricalActor(
                d_state=state_space.shape[0],
                d_hidden=d_hidden,
                d_action=action_space.shape[0],
                activation=activation,
            )
        else:
            self.pi = CategoricalActor(
                d_state=state_space.shape[0],
                d_action=int(action_space.n),
                d_hidden=d_hidden,
                activation=activation,
            )
        self.v = Critic(
            d_state=state_space.shape[0],
            d_hidden=d_hidden,
            activation=activation,
        )

    def step(self, state: Tensor) -> tuple[int, np.ndarray, float]:
        with torch.no_grad():
            policy: Categorical = self.pi.policy(state)
            action = policy.sample()
            logp_action = self.pi.logprob(policy, action)  # pyright: ignore[reportArgumentType]
            v = float(self.v(state))
            return (int(action), logp_action.numpy(), v)

    def act(self, state: Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.pi.policy(state).sample().numpy()


@final
class TrajectoryBuffer:
    def __init__(
        self,
        capacity: int,
        state_shape: list[int],
        action_shape: list[int],
        gamma: float = 0.99,
        lamda: float = 0.95,
    ):
        # env states
        self.states = np.zeros([capacity, *state_shape], dtype=np.float32)
        # predicted actions
        self.actions = np.zeros([capacity, *action_shape], dtype=np.int32)
        # action advantages
        self.advantages = np.zeros(capacity, dtype=np.float32)
        # action rewards
        self.rewards = np.zeros(capacity, dtype=np.float32)
        # discounted cumulative future rewards for the state
        self.returns = np.zeros(capacity, dtype=np.float32)
        # predicted env state values
        self.values = np.zeros(capacity, dtype=np.float32)
        # action log probabilities
        self.logps = np.zeros(capacity, dtype=np.float32)
        # discount factor
        self.gamma = gamma
        # value estimation discount
        self.lamda = lamda
        # index for the next insert
        self.next_index = 0
        # index for the start of the current episode
        self.episode_start_index = 0
        # buffer capacity
        self.capacity = capacity

    def push(
        self,
        state: np.ndarray,
        action: int,
        logp: float,
        value: float,
        reward: float,
    ):
        self.states[self.next_index] = state
        self.actions[self.next_index] = action
        self.logps[self.next_index] = logp
        self.values[self.next_index] = value
        self.rewards[self.next_index] = reward
        self.next_index += 1

    def push_episode_end(self, value: float, is_truncated: bool):
        range = slice(self.episode_start_index, self.next_index)
        ep_rewards = self.rewards[range]
        ep_values = self.values[range]

        # TD error
        bootstrap_value = value if is_truncated else 0.0
        next_values = np.append(ep_values[1:], bootstrap_value)
        deltas = ep_rewards + self.gamma * next_values - ep_values

        # GAE-Lambda advantage
        self.advantages[range] = cumulative_sum(deltas, self.gamma * self.lamda)

        # Return
        if is_truncated:
            ep_rewards = np.append(ep_rewards, bootstrap_value)
            self.returns[range] = cumulative_sum(ep_rewards, self.gamma)[:-1]
        else:
            self.returns[range] = cumulative_sum(ep_rewards, self.gamma)

        # Move the episode pointer
        self.episode_start_index = self.next_index

    def get_batch(self) -> "TrajectoryBatch":
        """TODO"""
        assert self.next_index == self.capacity
        self.next_index = 0
        self.episode_start_index = 0
        advantage_mean = np.mean(self.advantages)
        advantage_std = np.std(self.advantages)
        self.advantages = (self.advantages - advantage_mean) / advantage_std
        # print(f"trajectory actions: {self.actions}")
        return TrajectoryBatch(
            states=self.states,
            actions=self.actions,
            returns=self.returns,
            advantages=self.advantages,
            logps=self.logps,
        )


@dataclass
class TrajectoryBatch:
    states: np.ndarray
    actions: np.ndarray
    advantages: np.ndarray
    logps: np.ndarray
    returns: np.ndarray

    def as_torch(self) -> "TrajectoryTorchBatch":
        return TrajectoryTorchBatch(
            states=torch.as_tensor(self.states, dtype=torch.float32),
            actions=torch.as_tensor(self.actions, dtype=torch.long),
            advantages=torch.as_tensor(self.advantages, dtype=torch.float32),
            logps=torch.as_tensor(self.logps, dtype=torch.float32),
            returns=torch.as_tensor(self.returns, dtype=torch.float32),
        )


@dataclass
class TrajectoryTorchBatch:
    states: Tensor
    actions: Tensor
    advantages: Tensor
    logps: Tensor
    returns: Tensor


@dataclass
class PolicyInfo:
    approximate_kl: float
    mean_entropy: float
    clipped_fraction: float


def count_parameters(module: nn.Module) -> int:
    """Returns the total number of parameters in a NN module"""
    return int(sum(np.prod(p.shape) for p in module.parameters()))


def cumulative_sum(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    Returns the discounted cumulative sum of vector elements
    Example: cs([1,2,3], 0.95) => [5.59325, 4.835, 3]
    """
    result = np.empty_like(x, dtype=np.float32)
    result[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        result[i] = x[i] + gamma * result[i + 1]
    return result


if __name__ == "__main__":
    agent = CartPoleAgent()
    agent.train()
