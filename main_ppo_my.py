import numpy as np
from torch import nn
import torch

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

        features = self.conv(x)

        actor_distributions = torch.distributions.Categorical(probs=self.actor_head(features).softmax(-1))
        values = self.value_head(features).squeeze(1)
        return actor_distributions, values



class SnakePPOWrapper:
    def __init__(self, field_size):
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
            reward = 0
        if step_result == SnakeGame.SnakeGameActionResult.FOOD_EATEN:
            reward = 5
        if step_result == SnakeGame.SnakeGameActionResult.DEAD:
            reward = -1
            done = True
        if step_result == SnakeGame.SnakeGameActionResult.WON:
            reward = 10
            done = True

        self.n_steps += 1

        done = done or self.n_steps >= 10000
        if self.n_steps >= 10_000:
            print("PIZDAAAAAAAAAAAAAa")

        return reward, done


def main():
    model = PPONetwork().cuda()
    n_iterations = 10000000
    batch_size = 2048
    lr = 7e-4
    n_epochs = 8
    gamma = 0.95
    num_actions_to_collect = 4096
    epsilon = 0.2
    entropy_coefficient = 0.02
    return_coefficient = 5

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_episodes = 0
    for epoch in range(n_iterations):
        env = SnakePPOWrapper(16)

        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        old_log_probs = []
        model = model.eval()
        for game_iter in range(num_actions_to_collect):
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
                n_episodes += 1
                env = SnakePPOWrapper(16)

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
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.detach()
            policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()
            entropy_loss = -predicted_actions.entropy().mean()
            total_loss = policy_loss + loss_returns * return_coefficient + entropy_loss * entropy_coefficient

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()

        if epoch % 10 == 0:
            torch.save(model, "checkpoints_ppo_snake_my/checkpoint_0_0.pt")



















if __name__ == '__main__':
    main()
