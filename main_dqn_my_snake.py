import os.path
import random

import numpy as np
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter

import tb_utils
from snake_game import SnakeGame


def compute_returns(rewards, gamma, dones):
    result = np.zeros(len(rewards))
    cumulative_sum = 0
    for i in reversed(range(len(rewards))):
        if dones[i]:
            cumulative_sum = 0
        cumulative_sum = cumulative_sum * gamma + rewards[i]
        result[i] = cumulative_sum
    return result




class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        block = self.conv1(x)
        block = self.bn1(block)
        block = self.act1(block)

        return x + block


class DQNResidualNetwork4(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.LeakyReLU(inplace=True),

            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(inplace=True),

            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, 96, 3, 2, 1),
            nn.LeakyReLU(inplace=True),

            ResBlock(96),
            ResBlock(96),
        )

        self.position_encoding = nn.Parameter(
            torch.randn(1, 2, 16, 16)
        )
        nn.init.normal_(self.position_encoding, mean=0, std=0.05)

        self.value_head = nn.Sequential(
            ResBlock(96),

            nn.Conv2d(96, 96, 3, 2, 1),
            nn.LeakyReLU(inplace=True),

            nn.Flatten(1),
            nn.Linear(384, 384),
            nn.LeakyReLU(inplace=True),
            nn.Linear(384, 4),
        )

    def forward(self, x):
        assert x.ndim == 3
        x = x[:, None].float()

        x = torch.cat([x, self.position_encoding.tile(x.shape[0], 1, 1, 1)], dim=1)

        features = self.conv(x)

        values = self.value_head(features).squeeze(1)
        return values

    def forward_epsilon_greedy(self, state, epsilon):
        assert state.shape[0] == 1
        if random.random() < epsilon:
            return random.randrange(0, 4)

        with torch.inference_mode():
            return self(state).argmax().item()



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


class SnakePPOWrapper:
    def __init__(
            self,
            field_size=16, performed_reward=0, eaten_reward=10, dead_reward=-5, won_reward=100, terminate_iters=5000,
            n_steps_to_find_food: int = 1000, not_find_food_penalty: int = -0.1
    ):
        self.dead_reward = dead_reward
        self.eaten_reward = eaten_reward
        self.terminate_iters = terminate_iters
        self.performed_reward = performed_reward
        self.won_reward = won_reward
        self.game = SnakeGame(field_size, field_size)
        self.n_steps = 0
        self.n_steps_to_find_food = n_steps_to_find_food
        self.n_steps_without_food = 0
        self.not_find_food_penalty = not_find_food_penalty

        self.closest_distance = None

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
            reward = self.performed_reward
            self.n_steps_without_food += 1
        if step_result == SnakeGame.SnakeGameActionResult.FOOD_EATEN:
            reward = self.eaten_reward#  + min(self.game.score, 5)
            self.n_steps_without_food = 0
        if step_result == SnakeGame.SnakeGameActionResult.DEAD:
            reward = self.dead_reward
            done = True
        if step_result == SnakeGame.SnakeGameActionResult.WON:
            reward = self.won_reward
            done = True

        if self.n_steps_without_food >= self.n_steps_to_find_food:
            reward += self.not_find_food_penalty

        self.n_steps += 1

        done = done or self.n_steps >= self.terminate_iters
        if self.n_steps >= self.terminate_iters:
            print("Terminate")
        return reward, done


def main():
    seed = 0
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    writer = tb_utils.build_logger(
        "./logs_ppo_snake"
    )
    policy_net = QNetwork().cuda()
    target_net = QNetwork().cuda().requires_grad_(False)
    target_net.load_state_dict(policy_net.state_dict())

    n_iterations = 10000000
    batch_size = 128
    lr = 1e-4
    gamma = 0.94
    num_actions_to_collect = 4096

    random_epsilon_start = 10_000
    max_epsilon = 0.9
    min_epsilon = 0.005
    epsilon_decrease_epochs = 75000
    tau = 0.01

    env_params = {
        "field_size": 16,
        "performed_reward": -0.01,
        "eaten_reward": 3,
        "dead_reward": -1,
        "won_reward": 100,
        "terminate_iters": 5000,
        "n_steps_to_find_food": 50,
        "not_find_food_penalty": -0.1
    }
    hparam_dict = {
        "n_iterations": n_iterations,
        "batch_size": batch_size,
        "lr": lr,
        "gamma": gamma,
        "num_actions_to_collect": num_actions_to_collect,
        "model_class": str(policy_net.__class__),
    }
    hparam_dict.update(env_params)
    writer.add_hparams(hparam_dict, metric_dict={"default_hp": -1})

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=lr)# Самое последнее
    epsilon = max_epsilon
    n_episodes = 0
    for epoch in range(n_iterations):
        env = SnakePPOWrapper(**env_params)

        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        game_scores = []

        num_steps = []
        game_iter = 0
        while True:
            state_tensor = torch.from_numpy(env.game.field).cuda()[None]
            action_sample = policy_net.forward_epsilon_greedy(state_tensor, epsilon)
            states.append(state_tensor)
            actions.append(action_sample)

            reward, done = env.make_step(action_sample)
            next_states.append(torch.from_numpy(env.game.field).cuda()[None])
            rewards.append(reward)
            dones.append(done)

            if done:
                print(epoch, n_episodes, env.game.score, env.n_steps, epsilon)
                num_steps.append(env.n_steps)
                game_scores.append(env.game.score)
                n_episodes += 1
                env = SnakePPOWrapper(**env_params)

                epsilon = max(
                    max_epsilon - (n_episodes / epsilon_decrease_epochs) * (max_epsilon - min_epsilon), min_epsilon
                )
                if n_episodes > random_epsilon_start:
                    epsilon *= random.random()

            game_iter += 1
            if game_iter >= num_actions_to_collect and done:
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

        if epoch % 10 == 0:
            target_path = os.path.join(writer.log_dir, "Checkpoints/Checkpoint.pt")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            torch.save({"policy_net": policy_net.state_dict(), "target_net": target_net.state_dict(), "optimizer": optimizer.state_dict()}, target_path)

        writer.add_scalar("mean_score", np.mean(game_scores), epoch)
        writer.add_scalar("max_score", np.max(game_scores), epoch)
        writer.add_scalar("games_played", len(game_scores), epoch)
        writer.add_scalar("n_steps_mean", np.mean(num_steps), epoch)


if __name__ == '__main__':
    main()
