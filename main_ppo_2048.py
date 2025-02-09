import os.path
import random

import numpy as np
from torch import nn
import torch

import tb_utils
from game import ActionResult, Game2048


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


class PPOResidualNetwork4(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 3, 1, 1),
            nn.LeakyReLU(inplace=True),

            ResBlock(96),
            ResBlock(96),
            ResBlock(96),
            ResBlock(96),
            ResBlock(96),
            ResBlock(96),
            ResBlock(96),
            ResBlock(96),
            ResBlock(96),
        )

        self.actor_head = nn.Sequential(
            ResBlock(96),

            nn.Conv2d(96, 96, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            ResBlock(96),
            nn.Flatten(1),
            nn.Linear(384, 384),
            nn.LeakyReLU(inplace=True),
            nn.Linear(384, 4),
        )
        self.value_head = nn.Sequential(
            ResBlock(96),

            nn.Conv2d(96, 96, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            ResBlock(96),
            nn.Flatten(1),
            nn.Linear(384, 384),
            nn.LeakyReLU(inplace=True),
            nn.Linear(384, 1),
        )

    def forward(self, x):
        assert x.ndim == 3
        x = x[:, None].float()

        features = self.conv(x)

        actor_distributions = torch.distributions.Categorical(logits=self.actor_head(features))
        values = self.value_head(features).squeeze(1)
        return actor_distributions, values



class PPOTransformerNetwork(nn.Module):
    def __init__(self, field_size=4, d_model=128, n_heads=2, n_layers=5, dim_feedforward=512):
        super().__init__()

        self.input_projection = nn.Linear(1, d_model)
        self.position_encoding = nn.Parameter(torch.randn(1, field_size ** 2 + 2, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, batch_first=True), num_layers=n_layers
        )
        self.last_actor = nn.Linear(d_model, 4)
        self.last_value = nn.Linear(d_model, 1)

    def forward(self, x):
        assert x.ndim == 3
        inputs = x.flatten(1)[:, :, None]
        inputs[inputs == 0] = 1
        inputs = torch.log2(inputs)
        inputs = inputs / 7

        inputs = inputs * 2 - 1

        projected = self.input_projection(inputs)
        projected = torch.nn.functional.pad(projected, [0, 0, 0, 2])
        pe = self.position_encoding + projected
        tr = self.transformer(pe)
        last_actor = self.last_actor(tr[:, -2])
        last_value = self.last_value(tr[:, -1]).squeeze(-1)
        return torch.distributions.Categorical(logits=last_actor), last_value


class Game2048PPOWrapper:
    def __init__(self, field_size=4, four_prob=0.1):
        self.game = Game2048(field_size, four_prob)
        self.n_bad_steps = 0
        self.n_steps = 0
        self.score = 0

    def make_step(self, step_index):
        self.n_steps += 1
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
        merged_values = np.log2(merged_values)
        reward = 0
        for i in merged_values:
            if i < 8:
                reward += i
            else:
                reward += i * i

        done = False
        if step_result == ActionResult.ACTION_PERFORMED:
            reward += 0
        elif step_result == ActionResult.ACTION_BLOCKED:
            reward += -10
            self.n_bad_steps += 1
        else:
            done = True
            reward += -100

        reward /= 20

        done = done or self.n_bad_steps == 5

        self.score = np.sum(self.game.field)
        return reward, done

    @property
    def max_value(self):
        return np.max(self.game.field)



def main():
    seed = 0
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    writer = tb_utils.build_logger(
        "./logs_ppo_2048"
    )
    # model = PPOTransformerNetwork(d_model=256, n_layers=7, n_heads=4).cuda()
    model = torch.load("/home/valera/PycharmProjects/TwentyFourtyEight/logs_ppo_2048/run_28/Checkpoints/Checkpoint.pt")

    n_iterations = 10000000
    batch_size = 128
    lr = 3e-5
    n_epochs = 8 # Try a Different epoch count
    gamma = 0.95
    num_actions_to_collect = 4096
    epsilon = 0.2
    entropy_coefficient = 0.0001
    return_coefficient = 0.5

    env_params = {

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6)
    n_episodes = 0
    for epoch in range(3350, n_iterations):
        if epoch == 3370:
            print("Changing LR")
            for g in optimizer.param_groups:
                g['lr'] = lr
        env = Game2048PPOWrapper()

        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        old_log_probs = []
        game_scores = []
        num_steps = []
        game_max_scores = []
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
                print(epoch, n_episodes, env.score, env.n_steps, env.max_value)
                num_steps.append(env.n_steps)
                game_scores.append(env.score)
                n_episodes += 1
                game_max_scores.append(env.max_value)
                env = Game2048PPOWrapper()

            game_iter += 1
            if game_iter >= num_actions_to_collect and done:
                break

        returns = compute_returns(rewards, gamma, dones)

        states = torch.cat(states, 0)
        actions = torch.cat(actions, 0)
        returns = torch.from_numpy(returns).cuda().float()

        old_log_probs = torch.cat(old_log_probs, 0)
        model = model.train()

        for i in range(num_actions_to_collect * n_epochs // batch_size):
            samples_indices = torch.randint(0, states.shape[0], [batch_size])

            states_batch = states[samples_indices]
            actions_batch = actions[samples_indices]
            returns_batch = returns[samples_indices]
            old_log_probs_batch = old_log_probs[samples_indices]

            predicted_actions, predicted_returns = model(states_batch)
            new_log_probs = predicted_actions.log_prob(actions_batch)
            ratios = (new_log_probs - old_log_probs_batch).exp()

            loss_returns = nn.functional.l1_loss(predicted_returns, returns_batch)
            clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
            advantages = returns_batch - predicted_returns.detach()

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
            writer.add_scalar("returns_loss", loss_returns, epoch)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()

        if epoch % 10 == 0:
            target_path = os.path.join(writer.log_dir, "Checkpoints/Checkpoint.pt")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            torch.save(model, target_path)

        writer.add_scalar("mean_max_score", np.mean(game_max_scores), epoch)
        writer.add_scalar("max_max_score", np.max(game_max_scores), epoch)
        writer.add_scalar("min_max_score", np.min(game_max_scores), epoch)
        writer.add_scalar("mean_score", np.mean(game_scores), epoch)
        writer.add_scalar("max_score", np.max(game_scores), epoch)
        writer.add_scalar("games_played", len(game_scores), epoch)
        writer.add_scalar("n_steps_mean", np.mean(num_steps), epoch)


if __name__ == '__main__':
    main()
