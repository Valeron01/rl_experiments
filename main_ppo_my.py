import os.path

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


class PPONetwork(nn.Module):
    def __init__(self):
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
        # x = x / 3
        #
        # x[x==0] = -0.5

        features = self.conv(x)

        actor_distributions = torch.distributions.Categorical(probs=self.actor_head(features).softmax(-1))
        values = self.value_head(features).squeeze(1)
        return actor_distributions, values


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        block = self.conv1(x)
        # block = self.bn1(block)
        block = self.act1(block)

        block = self.conv2(block)
        # block = self.bn2(block)
        block = self.act2(block)

        return x + block


class PPOResidualNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.LeakyReLU(inplace=True),

            ResBlock(32),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(inplace=True),

            ResBlock(64),
            nn.Conv2d(64, 96, 3, 2, 1),
            nn.LeakyReLU(inplace=True),

            ResBlock(96),

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
        x = x / 3

        x[x==0] = -0.2

        features = self.conv(x)

        actor_distributions = torch.distributions.Categorical(probs=self.actor_head(features).softmax(-1))
        values = self.value_head(features).squeeze(1)
        return actor_distributions, values



class SnakePPOWrapper:
    def __init__(
            self,
            field_size=16, performed_reward=0, eaten_reward=10, dead_reward=-5, won_reward=100, terminate_iters=5000
    ):
        self.dead_reward = dead_reward
        self.eaten_reward = eaten_reward
        self.terminate_iters = terminate_iters
        self.performed_reward = performed_reward
        self.won_reward = won_reward
        self.game = SnakeGame(field_size, field_size)
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
            reward = self.performed_reward
        if step_result == SnakeGame.SnakeGameActionResult.FOOD_EATEN:
            reward = self.eaten_reward
        if step_result == SnakeGame.SnakeGameActionResult.DEAD:
            reward = self.dead_reward
            done = True
        if step_result == SnakeGame.SnakeGameActionResult.WON:
            reward = self.won_reward
            done = True

        self.n_steps += 1

        done = done or self.n_steps >= self.terminate_iters
        if self.n_steps >= self.terminate_iters:
            print("PIZDAAAAAAAAAAAAAa")

        return reward, done


def main():
    writer = tb_utils.build_logger(
        "./logs_ppo_snake"
    )
    model = PPONetwork().cuda()
    n_iterations = 10000000
    batch_size = 128
    lr = 3e-4
    n_epochs = 8
    gamma = 0.99
    num_actions_to_collect = 4096
    epsilon = 0.2
    entropy_coefficient = 0.01
    return_coefficient = 1

    env_params = {
        "field_size": 16,
        "performed_reward": -0.05,
        "eaten_reward": 10,
        "dead_reward": -5,
        "won_reward": 100,
        "terminate_iters": 5000
    }
    hparam_dict = {
        "n_iterations": n_iterations,
        "batch_size": batch_size,
        "lr": lr,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "num_actions_to_collect": num_actions_to_collect,
        "epsilon": epsilon,
        "entropy_coefficient": entropy_coefficient,
        "return_coefficient": return_coefficient,
        "model_class": str(model.__class__),
    }
    hparam_dict.update(env_params)
    writer.add_hparams(hparam_dict, metric_dict={"default_hp": -1})

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_episodes = 0
    for epoch in range(n_iterations):
        env = SnakePPOWrapper(**env_params)

        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        old_log_probs = []
        game_scores = []
        num_steps = []
        model = model.eval()
        game_iter = 0
        while True:
            state_tensor = torch.from_numpy(env.game.field).cuda()[None]
            with torch.no_grad():
                action, value = model(state_tensor)
            action_sample = action.sample()

            old_log_probs.append(action.log_prob(action_sample))
            states.append(state_tensor)
            actions.append(action_sample)

            reward, done = env.make_step(action_sample.item())
            rewards.append(reward)
            dones.append(done)
            values.append(value)

            if done:
                print(epoch, n_episodes, env.game.score, env.n_steps)
                num_steps.append(env.n_steps)
                game_scores.append(env.game.score)
                n_episodes += 1
                env = SnakePPOWrapper(**env_params)

            game_iter += 1
            if game_iter >= num_actions_to_collect and done:
                break

        returns = compute_returns(rewards, gamma, dones)

        states = torch.cat(states, 0)
        actions = torch.cat(actions, 0)
        returns = torch.from_numpy(returns).cuda()
        values = torch.cat(values, 0)

        old_log_probs = torch.cat(old_log_probs, 0)
        model = model.train()
        for i in range(num_actions_to_collect * n_epochs // batch_size):
            samples_indices = torch.randint(0, states.shape[0], [batch_size])

            states_batch = states[samples_indices]
            actions_batch = actions[samples_indices]
            returns_batch = returns[samples_indices]
            values_batch = values[samples_indices]
            old_log_probs_batch = old_log_probs[samples_indices]

            predicted_actions, predicted_returns = model(states_batch)
            new_log_probs = predicted_actions.log_prob(actions_batch)
            ratios = (new_log_probs - old_log_probs_batch).exp()

            loss_returns = nn.functional.smooth_l1_loss(predicted_returns, returns_batch)
            clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
            advantages = returns_batch - values_batch

            advantages_log = advantages.detach().mean()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.detach()
            policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()
            entropy_loss = -predicted_actions.entropy().mean()
            total_loss = policy_loss + loss_returns * return_coefficient + entropy_loss * entropy_coefficient

            writer.add_scalar("total_loss", total_loss, epoch)
            writer.add_scalar("policy_loss", policy_loss, epoch)
            writer.add_scalar("entropy_loss", entropy_loss, epoch)
            writer.add_scalar("advantages", advantages_log, epoch)


            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()

        if epoch % 10 == 0:
            target_path = os.path.join(writer.log_dir, "Checkpoints/Checkpoint.pt")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            torch.save(model, target_path)

        writer.add_scalar("mean_score", np.mean(game_scores), epoch)
        writer.add_scalar("max_score", np.max(game_scores), epoch)
        writer.add_scalar("games_played", len(game_scores), epoch)
        writer.add_scalar("n_steps_mean", np.mean(num_steps), epoch)


if __name__ == '__main__':
    main()
