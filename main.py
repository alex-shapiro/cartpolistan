import gymnasium as gym
import numpy as np


class CartPoleEvaluator:
    def __init__(self):
        self.eval_env: gym.Env[np.ndarray, int] = gym.make(
            "CartPole-v1",
            max_episode_steps=5000,
        )

    def select_action(self, state: np.ndarray) -> int:
        raise NotImplementedError

    def num_parameters(self) -> int:
        raise NotImplementedError

    def evaluate(self, n_episodes: int = 10):
        n_parameters = self.num_parameters()
        print(f"agent parameters: {n_parameters}")
        self.eval_env.reset()
        rewards: list[float] = []
        for i in range(n_episodes):
            state, _ = self.eval_env.reset()
            ended = False
            episode_reward = 0.0
            while not ended:
                action = self.select_action(state)
                state, reward, done, terminated, _ = self.eval_env.step(action)
                ended = done or terminated
                episode_reward += float(reward)
            rewards.append(episode_reward)
        r_mean = np.mean(rewards)
        r_std = np.std(rewards)
        print(f"Mean reward: {r_mean} +/- {r_std}")

        r_mean_pp = r_mean / n_parameters
        r_std_pp = r_std / n_parameters
        print(f"Mean reward per parameter: {r_mean_pp} +/- {r_std_pp}")


if __name__ == "__main__":
    evaluator = CartPoleEvaluator()
    evaluator.evaluate()
