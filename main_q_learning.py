import random

import torch
from torch import nn
import numpy as np
from tqdm import trange

from game import Game2048, ActionResult
from genetic_algorithm import Agent, GA2048Wrapper


class QNetwork(nn.Module):
    def __init__(self, field_size=4, d_model=128, n_heads=2, n_layers=5, dim_feedforward=512):
        super().__init__()

        self.input_projection = nn.Linear(1, d_model)
        self.position_encoding = nn.Parameter(torch.randn(1, field_size ** 2, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, batch_first=True), num_layers=n_layers
        )
        self.last = nn.Linear(d_model, 4)

    def forward(self, x):
        assert x.ndim == 3
        inputs = x.flatten(1)[:, :, None]
        inputs[inputs == 0] = 1
        inputs = torch.log2(inputs)
        inputs = inputs / 7

        inputs = inputs * 2 - 1

        projected = self.input_projection(inputs)
        pe = self.position_encoding + projected
        tr = self.transformer(pe)
        last = self.last(tr.mean(1))
        return last

    def forward_epsilon_greedy(self, state, epsilon):
        assert state.shape[0] == 1
        if random.random() < epsilon:
            return random.randrange(0, 4)

        return self(state).argmax().item()


class Game2048QWrapper:
    def __init__(self, field_size, four_prob=0.1):
        self.game = Game2048(field_size, four_prob)

    def make_step(self, step_index):
        # num_current_zeros = np.count_nonzero(self.game.field == 0)
        if step_index == 0:
            step_result, merged_values = self.game.move_top()
        elif step_index == 1:
            step_result, merged_values = self.game.move_right()
        elif step_index == 2:
            step_result, merged_values = self.game.move_bottom()
        elif step_index == 3:
            step_result, merged_values = self.game.move_left()
        else:
            raise NotImplementedError()

        # num_new_zeros = np.count_nonzero(self.game.field == 0)

        # delta_zeros = num_current_zeros - num_new_zeros

        reward = sum(np.power(np.log2(merged_values), 2))

        done = False
        if step_result == ActionResult.ACTION_PERFORMED:
            reward += 0
        elif step_result == ActionResult.ACTION_BLOCKED:
            reward += -5
        else:
            done = True
            reward += -100

        reward /= 100

        return reward, done


def main():
    max_epsilon = 0.8
    min_epsilon = 0.0005
    epsilon_decrease_epochs = 150000
    num_game_steps = 4096
    gamma = 0.95
    field_size = 4
    num_epochs = 10_000_000
    batch_size = 128
    lr = 1e-4
    weight_decay = 1e-2
    tau = 0.005

    model_parameters = {
        "field_size": field_size,
        "d_model": 256,
        "n_heads": 4,
        "dim_feedforward": 768,
        "n_layers": 6
    }

    policy_net = QNetwork(**model_parameters).cuda()
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    target_net = QNetwork(**model_parameters).cuda()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.requires_grad_(False)

    # loaded_target_net, loaded_policy_net, loaded_optimizer, _ = torch.load("./checkpoint4.pt")
    # target_net.load_state_dict(loaded_target_net.state_dict())
    # policy_net.load_state_dict(loaded_policy_net.state_dict())
    # optimizer.load_state_dict(loaded_optimizer.state_dict())
    max_value_per_game = []
    rewards_per_game = []
    epsilon_history = []
    for epoch in range(num_epochs):
        epsilon = max(max_epsilon - (epoch / epsilon_decrease_epochs) * (max_epsilon - min_epsilon), min_epsilon)
        game = Game2048QWrapper(field_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        game_reward = 0
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

            if done or len(states) == batch_size:
                states_batch = torch.cat(states, dim=0).cuda()
                actions_batch = torch.LongTensor(actions).cuda()
                rewards_batch = torch.FloatTensor(rewards).cuda()
                dones_batch = torch.FloatTensor(dones).cuda()
                next_states_batch = torch.cat(next_states, dim=0).cuda()

                predictions = policy_net(states_batch).gather(1, actions_batch[:, None]).squeeze(1)

                with torch.no_grad():
                    next_state_reward = target_net(next_states_batch).max(1).values * (1 - dones_batch)  # TARGET_NET

                target_predictions = next_state_reward * gamma + rewards_batch
                loss = nn.functional.smooth_l1_loss(predictions, target_predictions)
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

                states.clear()
                rewards.clear()
                actions.clear()
                next_states.clear()
                dones.clear()

            if done:
                break

        max_value_per_game.append(game.game.field.max())
        rewards_per_game.append(game_reward)
        epsilon_history.append(epsilon)
        print(game.game.field.max(), game_reward, epsilon, epoch)
        if epoch % 1000 == 0:
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
                },
                "max_value_per_game": max_value_per_game,
                "epsilon_history": epsilon_history,
                "rewards_per_game": rewards_per_game
            }, "./checkpoint2_1.pt")



















if __name__ == '__main__':
    main()
