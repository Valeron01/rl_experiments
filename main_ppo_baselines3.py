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


class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(4, 4), dtype=np.int32)
        self.game = Game2048(4, 0.1)
        self.n_bad_steps = 0

    def step(self, action):
        if action == 0:
            step_result, merged_values = self.game.move_top()
        elif action == 1:
            step_result, merged_values = self.game.move_right()
        elif action == 2:
            step_result, merged_values = self.game.move_bottom()
        elif action == 3:
            step_result, merged_values = self.game.move_left()
        else:
            raise NotImplementedError()

        reward = sum(np.power(np.log2(merged_values), 2))

        done = False
        if step_result == ActionResult.ACTION_PERFORMED:
            reward += 0
            self.n_bad_steps = 0
        elif step_result == ActionResult.ACTION_BLOCKED:
            reward += -15
            self.n_bad_steps += 1
        else:
            done = True
            reward += -500

        reward /= 50

        done = done or self.n_bad_steps == 5
        info = dict()

        if done:
            print("Max score: ", self.game.field.max())

        return self.game.field.copy(), reward, done, info

    def reset(self):
        self.game = Game2048(4, 0.1)
        self.n_bad_steps = 0
        return self.game.field.copy()  # reward, done, info can't be included

    def render(self, mode='human'):
        print(self.game.field)

    def close(self):
        ...


class PPONetwork(nn.Module):
    def __init__(self):
        super().__init__()
        field_size = 4
        d_model = 128
        n_heads = 2
        n_layers = 5
        dim_feedforward = 512

        # Shared components
        self.input_projection = nn.Linear(1, d_model)
        self.position_encoding = nn.Parameter(torch.randn(1, field_size ** 2, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, batch_first=True), num_layers=n_layers
        )

        # Policy head: Outputs logits for 4 actions (up, down, left, right)
        self.policy_head = nn.Linear(d_model, d_model)

    def forward(self, x):
        assert x.ndim == 3  # Input should be (batch_size, field_size, field_size)

        # Preprocess the input
        inputs = x.flatten(1)[:, :, None]  # Flatten the grid into (batch_size, field_size**2, 1)
        inputs[inputs == 0] = 1
        inputs = torch.log2(inputs)
        inputs = inputs / 7
        inputs = inputs * 2 - 1

        # Project input and add position encoding
        projected = self.input_projection(inputs)
        pe = self.position_encoding + projected

        # Pass through the transformer
        tr = self.transformer(pe)

        # Compute policy logits and value
        tr_mean = tr.mean(1)  # Take the mean along the sequence dimension
        policy_logits = self.policy_head(tr_mean)

        return policy_logits


class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, d_model=128):
        super().__init__(observation_space, features_dim=d_model)
        self.network = PPONetwork()

    def forward(self, observations):
        policy_logits = self.network(observations)
        return policy_logits




def main():
    # env = CustomEnv()

    # Function to create a single environment instance
    def make_env():
        def _init():
            return CustomEnv()

        return _init

    # Number of environments
    # num_envs = 32
    # Create the vectorized environment
    # env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    env = CustomEnv()

    # Assuming the observation space is 4x4
    ppo_model = PPO("MlpPolicy", env, policy_kwargs={"features_extractor_class": CustomExtractor}, device="cuda",
                    gamma=0.7, tensorboard_log="./log", n_epochs=4, n_steps=4096, ent_coef=10)
    ppo_model.learn(total_timesteps=1000000000000)


if __name__ == '__main__':
    main()
