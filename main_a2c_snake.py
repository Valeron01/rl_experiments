import random

import torch
from torch import nn
import numpy as np
from tqdm import trange

from game import Game2048, ActionResult
from genetic_algorithm import Agent, GA2048Wrapper
from snake_game import SnakeGame


class A2CNetwork(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()

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
            nn.Linear(384, 384),
            nn.LeakyReLU(inplace=True)
        )

        self.actor_head = nn.Linear(384, 4)
        self.value_head = nn.Linear(384, 1)

    def forward(self, x):
        assert x.ndim == 3
        x = x[:, None].float()

        features = self.conv(x)

        actor_distributions = torch.distributions.Categorical(probs=self.actor_head(features).softmax(-1))
        values = self.value_head(features).squeeze(1)
        return actor_distributions, values


class SnakeA2CWrapper:
    def __init__(self, field_size, four_prob=0.1):
        self.game = SnakeGame(16, 16)
        self.n_steps = 0

    def make_step(self, action):
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
            reward = 0
        if step_result == SnakeGame.SnakeGameActionResult.FOOD_EATEN:
            reward = 1.5
        if step_result == SnakeGame.SnakeGameActionResult.DEAD:
            reward = -1
            done = True
        if step_result == SnakeGame.SnakeGameActionResult.WON:
            reward = 10
            done = True

        self.n_steps += 1

        done = done or self.n_steps >= 10000
        if self.n_steps >= 10_000:
            print("Terminate")
            assert False

        return reward, done


def main():
    num_game_steps = 4096
    gamma = 0.99
    field_size = 16
    num_epochs = 10_000_000
    batch_size = 128
    lr = 1e-4
    weight_decay = 1e-2
    start_epoch = 0
    update_networks_every_n = 2048
    entropy_regularization = 0.5
    critic_weight = 5

    model_parameters = {
        "d_model": 128
    }

    policy_net = A2CNetwork(**model_parameters).cuda()
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=weight_decay)

    max_value_per_game = []
    rewards_per_game = []
    epsilon_history = []
    fields_per_game = []

    # if False:
    #     loaded_checkpoint = torch.load("checkpoints_dqn_snake/checkpoint2_0.pt")
    #     optimizer.load_state_dict(loaded_checkpoint["optimizer"])
    #     target_net.load_state_dict(loaded_checkpoint["target_net"])
    #     policy_net.load_state_dict(loaded_checkpoint["policy_net"])
    #     start_epoch = loaded_checkpoint["epoch"] + 1
    #     max_value_per_game = loaded_checkpoint["max_value_per_game"]
    #     rewards_per_game = loaded_checkpoint["rewards_per_game"]
    #     epsilon_history = loaded_checkpoint["epsilon_history"]
    #     fields_per_game = loaded_checkpoint["fields_per_game"]

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    for epoch in range(start_epoch, num_epochs):
        game = SnakeA2CWrapper(field_size)

        game_reward = 0

        for game_iter in range(num_game_steps):
            game_state = torch.from_numpy(game.game.field.copy())[None]
            policy_net = policy_net.eval()
            prediction, _ = policy_net(game_state.cuda())
            prediction = prediction.sample().item()
            reward, done = game.make_step(prediction)
            next_game_state = torch.from_numpy(game.game.field.copy())[None]

            game_reward += reward

            states.append(game_state)
            actions.append(prediction)
            rewards.append(reward)
            next_states.append(next_game_state)
            dones.append(done)

            if len(states) >= update_networks_every_n:
                print("UPDATING")
                states = torch.cat(states, dim=0).cuda()
                actions = torch.LongTensor(actions).cuda()
                rewards = torch.FloatTensor(rewards).cuda()
                dones = torch.FloatTensor(dones).cuda()
                next_states = torch.cat(next_states, dim=0).cuda()

                for _ in range(6):
                    batch_indices = torch.randperm(states.shape[0])
                    batch_indices = torch.tensor_split(batch_indices, max(1, batch_indices.shape[0] // batch_size))

                    policy_net.train()
                    for indices in batch_indices:
                        states_batch = states[indices]
                        actions_batch = actions[indices]
                        rewards_batch = rewards[indices]
                        dones_batch = dones[indices]
                        next_states_batch = next_states[indices]

                        predicted_actions, predicted_values = policy_net(states_batch)
                        with torch.no_grad():
                            _, predicted_future_values = policy_net(next_states_batch)

                        target_values = rewards_batch + gamma * predicted_future_values * (1 - dones_batch)

                        critic_loss = nn.functional.mse_loss(predicted_values, target_values)

                        advantage = rewards_batch + predicted_future_values * gamma - predicted_values.detach()
                        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                        actor_loss = (-advantage * predicted_actions.log_prob(actions_batch)).mean()

                        entropy_loss = -predicted_actions.entropy().mean()

                        total_loss = critic_loss * critic_weight + actor_loss + entropy_loss * entropy_regularization

                        optimizer.zero_grad()
                        total_loss.backward()
                        nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
                        optimizer.step()

                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []

            if done:
                print(game.game.score, game_reward, epoch, game_iter)
                break

        max_value_per_game.append(game.game.score)
        rewards_per_game.append(game_reward)
        fields_per_game.append(game.game.field)
        if epoch % 1000 == 0:
            print("Saving checkpoint")
            torch.save({
                "policy_net": policy_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "model_parameters": model_parameters,
                "hyper_parameters": {
                    "num_game_steps": num_game_steps,
                    "gamma": gamma,
                    "field_size": field_size,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "update_networks_every_n": update_networks_every_n
                },
                "max_value_per_game": max_value_per_game,
                "epsilon_history": epsilon_history,
                "rewards_per_game": rewards_per_game,
                "fields_per_game": fields_per_game
            }, "checkpoints_a2c_snake/checkpoint5_1.pt")


if __name__ == '__main__':
    main()
