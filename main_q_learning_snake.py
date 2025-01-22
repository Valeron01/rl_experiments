import random

import torch
from torch import nn
import numpy as np
from tqdm import trange

from game import Game2048, ActionResult
from genetic_algorithm import Agent, GA2048Wrapper
from snake_game import SnakeGame


class QNetwork(nn.Module):
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
            nn.Linear(384, d_model),
            nn.LeakyReLU(inplace=True),
            nn.Linear(d_model, 4)
        )

    def forward(self, x):
        assert x.ndim == 3
        x = x[:, None].float()
        return self.conv(x)

    def forward_epsilon_greedy(self, state, epsilon):
        assert state.shape[0] == 1
        if random.random() < epsilon:
            return random.randrange(0, 4)

        with torch.inference_mode():
            return self(state).argmax().item()


class SnakeQWrapper:
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
            reward = -0.001
        if step_result == SnakeGame.SnakeGameActionResult.FOOD_EATEN:
            reward = 10
        if step_result == SnakeGame.SnakeGameActionResult.DEAD:
            reward = -5
            done = True
        if step_result == SnakeGame.SnakeGameActionResult.WON:
            reward = 5
            done = True

        self.n_steps += 1

        done = done or self.n_steps >= 10000

        return reward, done


def main():
    max_epsilon = 0.9
    min_epsilon = 0.005
    epsilon_decrease_epochs = 75000
    num_game_steps = 4096
    gamma = 0.9
    field_size = 16
    num_epochs = 10_000_000
    batch_size = 128
    lr = 3e-4
    weight_decay = 1e-2
    tau = 0.01
    start_epoch = 0
    random_epsilon_start = 10_000

    model_parameters = {
        "d_model": 128
    }

    policy_net = QNetwork(**model_parameters).cuda()
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    target_net = QNetwork(**model_parameters).cuda()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.requires_grad_(False)

    max_value_per_game = []
    rewards_per_game = []
    epsilon_history = []
    fields_per_game = []

    if True:
        loaded_checkpoint = torch.load("checkpoints_dqn_snake/checkpoint2_0.pt")
        optimizer.load_state_dict(loaded_checkpoint["optimizer"])
        target_net.load_state_dict(loaded_checkpoint["target_net"])
        policy_net.load_state_dict(loaded_checkpoint["policy_net"])
        start_epoch = loaded_checkpoint["epoch"] + 1
        max_value_per_game = loaded_checkpoint["max_value_per_game"]
        rewards_per_game = loaded_checkpoint["rewards_per_game"]
        epsilon_history = loaded_checkpoint["epsilon_history"]
        fields_per_game = loaded_checkpoint["fields_per_game"]

    for epoch in range(start_epoch, num_epochs):
        epsilon = max(
            max_epsilon - (epoch / epsilon_decrease_epochs) * (max_epsilon - min_epsilon), min_epsilon
        )
        if epoch > random_epsilon_start:
            epsilon *= random.random()

        game = SnakeQWrapper(field_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        game_reward = 0
        policy_net = policy_net.eval()
        for game_iter in range(num_game_steps):
            game_state = torch.from_numpy(game.game.field.copy())[None]
            prediction = policy_net.forward_epsilon_greedy(game_state.cuda(), epsilon)
            reward, done = game.make_step(prediction)
            next_game_state = torch.from_numpy(game.game.field.copy())[None]

            game_reward += reward

            states.append(game_state)
            actions.append(prediction)
            rewards.append(reward)
            next_states.append(next_game_state)
            dones.append(done)

            if done:
                break

        states = torch.cat(states, dim=0).cuda()
        actions = torch.LongTensor(actions).cuda()
        rewards = torch.FloatTensor(rewards).cuda()
        dones = torch.FloatTensor(dones).cuda()
        next_states = torch.cat(next_states, dim=0).cuda()

        batch_indices = torch.randperm(states.shape[0])
        batch_indices = torch.tensor_split(batch_indices, max(1, batch_indices.shape[0] // batch_size))

        policy_net.train()
        target_net.eval()
        for indices in batch_indices:
            states_batch = states[indices]
            actions_batch = actions[indices]
            rewards_batch = rewards[indices]
            dones_batch = dones[indices]
            next_states_batch = next_states[indices]

            predictions = policy_net(states_batch).gather(1, actions_batch[:, None]).squeeze(1)

            with torch.no_grad():
                next_state_reward = target_net(next_states_batch).max(1).values * (1 - dones_batch)  # TARGET_NET

            target_predictions = next_state_reward * gamma + rewards_batch
            loss = nn.functional.smooth_l1_loss(predictions, target_predictions) * states_batch.shape[0] / batch_size
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(policy_net.parameters(), 10)
            optimizer.step()

            with torch.no_grad():
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (
                            1 - tau)
                target_net.load_state_dict(target_net_state_dict)

        max_value_per_game.append(game.game.score)
        rewards_per_game.append(game_reward)
        epsilon_history.append(epsilon)
        fields_per_game.append(game.game.field)
        print(game.game.score, game_reward, epsilon, epoch)
        if epoch % 1000 == 0:
            print("Saving checkpoint")
            torch.save({
                "target_net": target_net.state_dict(),
                "policy_net": policy_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "model_parameters": model_parameters,
                "hyper_parameters": {
                    "max_epsilon": max_epsilon,
                    "min_epsilon": min_epsilon,
                    "epsilon_decrease_epochs": epsilon_decrease_epochs,
                    "num_game_steps": num_game_steps,
                    "gamma": gamma,
                    "field_size": field_size,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "tau": tau,
                    "random_epsilon_start": random_epsilon_start
                },
                "max_value_per_game": max_value_per_game,
                "epsilon_history": epsilon_history,
                "rewards_per_game": rewards_per_game,
                "fields_per_game": fields_per_game
            }, "checkpoints_dqn_snake/checkpoint2_1.pt")


if __name__ == '__main__':
    main()
