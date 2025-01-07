import random

import torch
from torch import nn
import numpy as np
from tqdm import trange

from game import Game2048, ActionResult
from genetic_algorithm import Agent, GA2048Wrapper


class QNetwork(nn.Module):
    def __init__(self, field_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(field_size * field_size, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        assert x.ndim == 3
        inputs = x.flatten(1)
        inputs[inputs == 0] = 1
        inputs = torch.log2(inputs)
        inputs = inputs / 5

        inputs = inputs * 2 - 1
        return self.network(inputs)

    def forward_epsilon_greedy(self, state, epsilon):
        assert state.shape[0] == 1
        if random.random() < epsilon:
            return random.randrange(0, 4)

        return self(state).argmax().item()


class Game2048QWrapper:
    def __init__(self, field_size, four_prob=0.1):
        self.game = Game2048(field_size, four_prob)

    def make_step(self, step_index):
        num_current_zeros = np.count_nonzero(self.game.field == 0)
        if step_index == 0:
            step_result = self.game.move_top()
        elif step_index == 1:
            step_result = self.game.move_right()
        elif step_index == 2:
            step_result = self.game.move_bottom()
        elif step_index == 3:
            step_result = self.game.move_left()
        else:
            raise NotImplementedError()

        num_new_zeros = np.count_nonzero(self.game.field == 0)

        delta_zeros = num_current_zeros - num_new_zeros

        reward = delta_zeros + 1
        reward *= 2
        done = False
        if step_result == ActionResult.ACTION_PERFORMED:
            reward += 1
        elif step_result == ActionResult.ACTION_BLOCKED:
            reward += -1
        else:
            done = True
            reward += -15

        return reward, done


def main():
    epsilon = 1
    epsilon_decay = 0.999
    min_epsilon = 0.05
    num_game_steps = 1536
    gamma = 0.95
    field_size = 4
    num_epochs = 10000
    batch_size = 32
    lr = 1e-3
    weight_decay = 1e-2
    tau = 0.005

    policy_net = QNetwork(field_size)
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    target_net = QNetwork(field_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.requires_grad_(False)

    for epoch in range(num_epochs):
        game = Game2048QWrapper(field_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for game_iter in range(num_game_steps):
            game_state = torch.from_numpy(game.game.field.copy())[None]
            prediction = policy_net.forward_epsilon_greedy(game_state, epsilon)
            reward, done = game.make_step(prediction)
            next_game_state = torch.from_numpy(game.game.field.copy())[None]

            states.append(game_state)
            actions.append(prediction)
            rewards.append(reward)
            next_states.append(next_game_state)
            dones.append(done)

            if done or len(states) == batch_size:
                pass

            if done:
                break
        assert len(dones) != 0
        epsilon = max(epsilon * epsilon_decay, min_epsilon)

        states = torch.cat(states, dim=0)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.cat(next_states, dim=0)
        dones = torch.FloatTensor(dones)

        shuffle_indices = torch.randperm(len(dones))
        batched_indices = torch.tensor_split(shuffle_indices, max(1, len(shuffle_indices) // batch_size))

        for batch_indices in batched_indices:
            states_batch = states[batch_indices]
            actions_batch = actions[batch_indices]
            rewards_batch = rewards[batch_indices]
            dones_batch = dones[batch_indices]
            next_states_batch = next_states[batch_indices]

            predictions = policy_net(states_batch).gather(1, actions_batch[:, None]).squeeze(1)

            with torch.no_grad():
                next_state_reward = target_net(next_states_batch).max(1).values * (1 - dones_batch) # TARGET_NET

            target_predictions = next_state_reward * gamma + rewards_batch
            loss = nn.functional.smooth_l1_loss(predictions, target_predictions)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            optimizer.step()
            print(len(dones))

            with torch.no_grad():
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (
                                1 - tau)
                target_net.load_state_dict(target_net_state_dict)















if __name__ == '__main__':
    main()
