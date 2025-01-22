import gym
import numpy as np
import torch
from gym import spaces
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from stable_baselines3.common.vec_env import SubprocVecEnv

from game import Game2048, ActionResult
from snake_game import SnakeGame


class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(16, 16), dtype=np.uint8)
        self.game = SnakeGame(16, 16)
        self.n_steps = 0

    def step(self, action):
        if action == 0:
            step_result = self.game.move_up()
        elif action == 1:
            step_result = self.game.move_right()
        elif action == 2:
            step_result = self.game.move_down()
        elif action == 3:
            step_result = self.game.move_left()
        else:
            raise NotImplementedError()

        reward = 0
        done = False
        if step_result == SnakeGame.SnakeGameActionResult.ACTION_PERFORMED:
            reward = -0.001
        if step_result == SnakeGame.SnakeGameActionResult.FOOD_EATEN:
            reward = 1
        if step_result == SnakeGame.SnakeGameActionResult.DEAD:
            reward = -1
            done = True
            print(self.game.score)
        if step_result == SnakeGame.SnakeGameActionResult.WON:
            reward = 5
            done = True

        self.n_steps += 1

        done = done or self.n_steps >= 1000

        return self.game.field.copy(), reward, done, {}

    def reset(self):
        self.game.reset()
        self.n_steps = 0
        return self.game.field.copy()  # reward, done, info can't be included

    def render(self, mode='human'):
        print(self.game.field)

    def close(self):
        ...


class PPONetwork(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 128

        # Shared components
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 96, 3, 2, 1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(96, 96, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(1),
            nn.Linear(384, d_model)
        )

    def forward(self, x):
        assert x.ndim == 3  # Input should be (batch_size, field_size, field_size)

        x = x[:, None]
        return self.conv(x)


class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, d_model=128):
        super().__init__(observation_space, features_dim=d_model)
        self.network = PPONetwork()

    def forward(self, observations):
        policy_logits = self.network(observations)
        return policy_logits


def main():
    env = CustomEnv()

    # Assuming the observation space is 4x4
    ppo_model = PPO("MlpPolicy", env, policy_kwargs={"features_extractor_class": CustomExtractor}, device="cuda",
                    gamma=0.95, tensorboard_log="./log", n_epochs=8, n_steps=1024, ent_coef=0.01)
    ppo_model.learn(total_timesteps=1000000000000)


if __name__ == '__main__':
    main()
