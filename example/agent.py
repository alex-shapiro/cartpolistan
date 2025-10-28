import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import final, override

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import RecordVideo
from numpy import ndarray
from torch import Tensor, nn
from torch.optim import Adam

from .replay_buffer import ReplayBuffer


@final
class CartPoleAgent:
    def __init__(self, seed: int = 2023):
        # simulation environment
        self.env: gym.Env[ndarray, int] = gym.make("CartPoleCustom-v0")  # pyright: ignore[reportUnknownMemberType]
        self.eval_env: gym.Env[ndarray, int] = gym.make(  # pyright: ignore[reportUnknownMemberType]
            "CartPoleCustom-v0"
        )

        self.state_space: Box = self.env.observation_space  # pyright: ignore[reportAttributeAccessIssue]
        self.action_space: Discrete = self.env.action_space  # pyright: ignore[reportAttributeAccessIssue]

        # initialize random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
        self.env.action_space.seed(seed)  # pyright: ignore[reportUnusedCallResult]
        self.seed = seed

        # replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=200_000,
            state_space=self.state_space,
            action_space=self.action_space,
        )

        # exploration hyperparameters
        self.num_steps = 160_000
        self.eps_initial = 1.0
        self.eps_final = 0.02
        self.exploration_fraction = 0.05

        # other hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        self.batch_size = 64

        # models
        self.state_size = self.state_space.shape[0]
        self.action_size = int(self.action_space.n)
        self.perf_timers = PerfTimers()

        self.device = torch.device("cpu")
        self.q_net = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_net = QNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train(
        self,
        model_update_interval: int = 5,
        target_update_interval: int = 1000,
        eval_interval: int = 10000,
        num_eval_episodes: int = 100,
    ):
        state, _ = self.env.reset(seed=self.seed)

        for step in range(1, self.num_steps + 1):
            start = time.perf_counter()
            eps = self.exploration_rate(step)
            state = self.step(state, eps)

            if step % target_update_interval == 0:
                _ = self.target_net.load_state_dict(self.q_net.state_dict())

            if step % model_update_interval == 0 and step > 1000:
                self.replay()

            self.perf_timers.overall.append(time.perf_counter() - start)

            if step % eval_interval == 0:
                episode_rewards = self.eval(n_episodes=num_eval_episodes)
                r_mean = np.mean(episode_rewards)
                r_std = np.std(episode_rewards)
                print()
                print(f"Evaluation at step {step}:")
                print(f"- eps: {eps:.2f}")
                print(f"- reward: {r_mean:.2f} +/- {r_std:.2f}")
                state, _ = self.env.reset()

    def step(self, state: ndarray, eps: float) -> ndarray:
        """Runs the next step and returns the resulting state"""
        start = time.perf_counter()
        action = self.epsilon_greedy_action(state, eps, self.q_net)
        next_state, reward, terminated, truncated, _info = self.env.step(action)
        self.replay_buffer.push(
            state=state,
            next_state=next_state,
            action=action,
            reward=float(reward),
            terminated=(terminated or truncated),
        )

        # reset the env if the episode has ended
        if terminated or truncated:
            next_state, _ = self.env.reset()

        self.perf_timers.step.append(time.perf_counter() - start)
        return next_state

    def epsilon_greedy_action(
        self,
        state: ndarray,
        eps: float,
        q_net: "QNetwork",
    ) -> int:
        start = time.perf_counter()
        if np.random.rand() <= eps:
            action = int(self.action_space.sample())
        else:
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, device=self.device).unsqueeze(0)
                q_values: Tensor = q_net(state_tensor)  # pyright: ignore[reportAny]
                action = int(q_values.argmax().item())
        self.perf_timers.eps_greedy_action.append(time.perf_counter() - start)
        return action

    def replay(self):
        start = time.perf_counter()

        # fetch N random samples from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size).to_torch(self.device)

        # Predict max Q values across all possible actions in the next state
        # Bootstrap iff the state is nonterminal; otherwise td_target is just the reward
        with torch.no_grad():
            next_q_values: Tensor = self.target_net(batch.next_states)  # pyright: ignore[reportAny]
            next_q_values, _ = next_q_values.max(dim=1)
            should_bootstrap = batch.terminateds.logical_not()
            td_target = (
                batch.rewards + self.discount_factor * next_q_values * should_bootstrap
            )

        # Predict Q values for all possible actions in a state
        # Then take ("gather") just the Q value for the selected action
        current_q_values: Tensor = self.q_net(batch.states)  # pyright: ignore[reportAny]
        current_q_values = current_q_values.gather(1, batch.actions)
        current_q_values = current_q_values.squeeze(dim=-1)

        # sanity check
        assert current_q_values.shape == (self.batch_size,)
        assert current_q_values.shape == td_target.shape

        # Compute MSE loss & backprop
        loss = ((current_q_values - td_target) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()  # pyright: ignore [reportUnknownMemberType, reportUnusedCallResult]
        self.optimizer.step()  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
        self.perf_timers.replay.append(time.perf_counter() - start)

    def eval(
        self, n_episodes: int, eps: float = 0.0, video_name: str | None = None
    ) -> list[float]:
        """Evaluates the model"""
        episode_rewards: list[float] = []

        video_path = None
        if video_name is None:
            eval_env = self.eval_env
        else:
            video_path = Path(__file__).parent / "logs" / "videos" / video_name
            video_path.parent.mkdir(parents=True, exist_ok=True)
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="gymnasium.wrappers.rendering"
            )
            eval_env = RecordVideo(  # pyright: ignore[reportUnknownVariableType]
                self.eval_env,
                str(video_path.parent),
                step_trigger=lambda _: False,
                video_length=100_000,
            )
            eval_env.start_recording(video_name)

        for _ in range(n_episodes):
            total_reward = 0.0
            state, _ = eval_env.reset()
            done = False
            while not done:
                action = self.epsilon_greedy_action(state, eps, self.q_net)
                state, reward, terminated, truncated, _info = eval_env.step(action)
                total_reward += float(reward)
                done = terminated or truncated
            episode_rewards.append(total_reward)

        if video_path is not None:
            eval_env.close()

        return episode_rewards

    def exploration_rate(self, step: int) -> float:
        """Returns the current value of the exploration rate according to a linear schedule"""
        explore_steps = self.num_steps * self.exploration_fraction
        progress = min(step / explore_steps, 1.0)
        return self.eps_initial + progress * (self.eps_final - self.eps_initial)


@final
class QNetwork(nn.Module):
    """
    NN for Q-Value prediction
    Has 3 fully connected layers separated by ReLU activations
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super(QNetwork, self).__init__()  # pyright: ignore[reportUnknownMemberType]
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    @override
    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)  # pyright: ignore[reportAny]


@dataclass
class PerfTimers:
    overall: list[float] = field(default_factory=list)
    step: list[float] = field(default_factory=list)
    replay: list[float] = field(default_factory=list)
    eps_greedy_action: list[float] = field(default_factory=list)

    @override
    def __repr__(self) -> str:
        mean_overall = np.mean(self.overall)
        return "\n".join(
            [
                "Perf Timers:",
                f"- overall: {mean_overall:.4f} sec",
                f"- step(): {np.mean(self.step) / mean_overall:.2f}",
                f"- replay(): {np.mean(self.replay) / mean_overall:.2f}",
                f"- eps_greedy_action(): {np.mean(self.eps_greedy_action) / mean_overall:.2f}",
            ]
        )


if __name__ == "__main__":
    agent = CartPoleAgent()
    agent.train()
