import gymnasium as gym
import numpy as np

import eval_env


class CartPoleEvaluator:
    def __init__(self):
        self.headless_env: gym.Env[np.ndarray, int] = gym.make(
            "CartPoleCustom-v0",
            max_episode_steps=5000,
        )
        self.rendered_env: gym.Env[np.ndarray, int] = gym.make(
            "CartPoleCustom-v0",
            max_episode_steps=5000,
            render_mode="human",
        )
        # TODO: init your agent here

    def select_action(self, state: np.ndarray) -> int:
        # TODO: implement me!
        raise NotImplementedError

    def num_parameters(self) -> int:
        # TODO: implement me!
        raise NotImplementedError

    def evaluate(self, n_episodes: int = 10):
        n_parameters = self.num_parameters()
        print(f"agent parameters: {n_parameters}")
        rewards: list[float] = []
        for i in range(n_episodes):
            env = self.rendered_env
            state, _ = env.reset()
            ended = False
            episode_reward = 0.0
            while not ended:
                action = self.select_action(state)
                state, reward, done, terminated, _ = env.step(action)
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
