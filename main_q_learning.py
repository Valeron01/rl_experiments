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

        self.input_projection = nn.Linear(1, 128)
        self.position_encoding = nn.Parameter(torch.randn(1, 16, 128))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128, 2, 512, batch_first=True), num_layers=5
        )
        self.last = nn.Linear(128, 4)

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

        reward = sum(np.square(np.log2(merged_values)))

        done = False
        if step_result == ActionResult.ACTION_PERFORMED:
            reward += 0
        elif step_result == ActionResult.ACTION_BLOCKED:
            reward += -5
        else:
            done = True
            reward += -50

        reward /= 10

        return reward, done


def main():
    epsilon = 0.9
    min_epsilon = 0.05
    epsilon_decrease_epochs = 20_000
    num_game_steps = 1536
    gamma = 0.95
    field_size = 4
    num_epochs = 10_000_000
    batch_size = 128
    lr = 1e-4
    weight_decay = 1e-2
    tau = 0.005

    epsilon_delta = (epsilon - min_epsilon) / epsilon_decrease_epochs

    policy_net = QNetwork(field_size).cuda()
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    target_net = QNetwork(field_size).cuda()
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
            prediction = policy_net.forward_epsilon_greedy(game_state.cuda(), epsilon)
            reward, done = game.make_step(prediction)
            next_game_state = torch.from_numpy(game.game.field.copy())[None]

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
        if epoch % 1000 == 0:
            torch.save([target_net, policy_net, optimizer, epoch], "./checkpoint4.pt")
        print(game.game.field.max(), epsilon, epoch)
        epsilon = max(epsilon - epsilon_delta, min_epsilon)



















if __name__ == '__main__':
    main()
