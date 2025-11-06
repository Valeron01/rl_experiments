import os.path

import numpy as np
import torch
from torch import nn
from tqdm import trange

import tb_utils
from environments.minesweeper.minesweeper_env import MinesweeperEnv


class MinesweeperConvModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(2, 16, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )

        self.down_1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.down_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.down_3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )

        self.mid = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )

        self.up_3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )

        self.up_2 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )

        self.up_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )

        self.last = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 1, 3, 1, 1)
        )

    def forward(self, x):
        stem = self.stem(x)

        down_1 = self.down_1(stem)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)

        mid = self.mid(down_3)

        up_3 = self.up_3(mid)
        up_2 = self.up_2(up_3 + down_2)
        up_1 = self.up_1(up_2 + down_1)

        logits = self.last(up_1 + stem).flatten(1)
        return torch.distributions.Categorical(logits=logits, validate_args=True)


def main():
    device = "cuda:0"
    env_params = {
        "height": 16,
        "width": 32,
        "num_bombs": 80,
        "lost_reward": -0.001,
        "opened_reward": 0.005,
        "wrong_move_reward": -0.001,
        "score_multiplier": 0.01
    }
    writer = tb_utils.build_logger(
        "./logs_reinforce_minesweeper"
    )
    checkpoint = torch.load("./logs_reinforce_minesweeper/run_063/Checkpoints/Checkpoint.pt")
    lr = 5e-4
    entropy_coefficient = 1e-8
    n_samples_per_update = 1536
    n_epochs_per_update = 2
    batch_size = 64
    model = MinesweeperConvModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    printer = trange(23180, 100000)
    for epoch in printer:
        environment = MinesweeperEnv(**env_params)
        scores = []
        model.eval()
        actions = []
        rewards = []
        states = []
        entropies = []
        max_probs = []
        for i in range(n_samples_per_update):
            state = torch.from_numpy(environment.state())[None].to(device)
            states.append(state)
            with torch.no_grad():
                decision = model(state)
            entropies.append(decision.entropy().item())
            action = decision.sample().item()
            reward, done = environment.step(action)
            actions.append(action)
            rewards.append(reward)
            max_probs.append(decision.probs.max().item())

            if done:
                scores.append(environment.score())
                environment = MinesweeperEnv(**env_params)

        scores.append(environment.score())

        states = torch.cat(states, 0)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        model.train()

        for _ in range(n_epochs_per_update):
            random_indices = torch.randperm(n_samples_per_update).view(-1, batch_size)
            for update_iter in range(n_samples_per_update // batch_size):
                batch_samples = random_indices[update_iter]

                sampled_actions = actions[batch_samples]
                sampled_rewards = rewards[batch_samples]
                sampled_states = states[batch_samples]

                results = model(sampled_states)
                log_prob = results.log_prob(sampled_actions)

                loss = -(log_prob * sampled_rewards).mean() + results.entropy().mean() * entropy_coefficient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        writer.add_scalar("mean_score", np.mean(scores), epoch)
        writer.add_scalar("max_score", np.max(scores), epoch)
        writer.add_scalar("min_score", np.min(scores), epoch)

        writer.add_scalar("mean_entropy", np.mean(entropies), epoch)
        writer.add_scalar("max_entropy", np.max(entropies), epoch)
        writer.add_scalar("min_entropy", np.min(entropies), epoch)

        writer.add_scalar("max_prob", np.max(max_probs), epoch)
        writer.add_scalar("mean_reward", rewards.mean(), epoch)

        printer.set_postfix({
            "mean_score": np.mean(scores),
            "max_score": np.max(scores),
            "min_score": np.min(scores),
        })

        if epoch % 10 == 0:
            target_path = os.path.join(writer.log_dir, "Checkpoints/Checkpoint.pt")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "env_params": env_params
                },
                target_path
            )


if __name__ == '__main__':
    main()
