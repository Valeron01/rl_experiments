import gym
import torch
from gym import spaces
import numpy as np
from torch import optim, nn
from torch.distributions import Categorical
from tqdm import trange


class PPONetwork(nn.Module):
    def __init__(self, field_size=4, d_model=128, n_heads=2, n_layers=5, dim_feedforward=512):
        super().__init__()

        self.input_projection = nn.Linear(1, d_model)
        self.position_encoding = nn.Parameter(torch.randn(1, field_size ** 2, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, batch_first=True), num_layers=n_layers
        )
        self.actor_head = nn.Linear(d_model, 4)
        self.critic_head = nn.Linear(d_model, 1)

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
        features = tr.mean(1)
        actor_head = self.actor_head(features).softmax(-1)
        critic_head = self.critic_head(features)


        return actor_head, critic_head




class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        self.board_size = 4
        self.action_space = spaces.Discrete(4)  # Actions: 0=up, 1=down, 2=left, 3=right
        self.observation_space = spaces.Box(
            low=0, high=2 ** 16, shape=(self.board_size, self.board_size), dtype=np.int32
        )
        self.board = None

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self._add_new_tile()
        self._add_new_tile()
        return self.board  # Return state as a flat vector

    def step(self, action):
        if self._is_valid_action(action):
            self.board = self._move(action)
            self._add_new_tile()
            reward = self._calculate_reward() / 1024
            done = not self._can_make_any_move()
            return self.board, reward, done, {}
        else:
            # Invalid action: Penalize and do not change state
            return self.board, -1, False, {}

    def render(self, mode="human"):
        print(self.board)

    def _add_new_tile(self):
        empty_cells = np.argwhere(self.board == 0)
        if empty_cells.size > 0:
            row, col = empty_cells[np.random.choice(len(empty_cells))]
            self.board[row, col] = 2 if np.random.rand() < 0.9 else 4

    def _is_valid_action(self, action):
        # Check if action leads to a state change
        return not np.array_equal(self.board, self._move(action))

    def _move(self, action):
        def slide_and_merge(row):
            row = row[row != 0]  # Remove zeros
            for i in range(len(row) - 1):  # Merge tiles
                if row[i] == row[i + 1]:
                    row[i] *= 2
                    row[i + 1] = 0
            row = row[row != 0]  # Remove zeros again
            return np.pad(row, (0, self.board_size - len(row)), constant_values=0)

        rotated_board = np.rot90(self.board, -action)  # Rotate for movement
        moved_board = np.array([slide_and_merge(row) for row in rotated_board])
        return np.rot90(moved_board, action)  # Rotate back

    def _calculate_reward(self):
        return np.sum(self.board)  # Reward: Total tile values

    def _can_make_any_move(self):
        for action in range(4):
            if self._is_valid_action(action):
                return True
        return False


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []



buffer = RolloutBuffer()


GAMMA = 0.99           # Discount factor
EPSILON_CLIP = 0.2     # Clipping range for PPO
LR = 0.001             # Learning rate
EPOCHS = 10            # Number of training epochs per update
BATCH_SIZE = 64        # Mini-batch size
STEPS_PER_UPDATE = 2048  # Steps before updating the policy


def compute_gae(rewards, values, dones, gamma=GAMMA, lambda_=0.95):
    advantages = []
    gae = 0
    next_value = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = delta + gamma * lambda_ * (1 - dones[step]) * gae
        advantages.insert(0, gae)
        next_value = values[step]
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(values, dtype=torch.float32)
    return advantages.cuda(), returns.cuda()


env = Game2048Env()
input_size = env.observation_space.shape[0]  # Flattened board size
action_size = env.action_space.n
model = PPONetwork(4, 128, 2).cuda()
optimizer = optim.Adam(model.parameters(), lr=LR)

buffer = RolloutBuffer()

# Training loop
for episode in range(1000):
    state = env.reset()
    buffer.clear()
    total_reward = 0

    # Collect rollout data
    for _ in range(STEPS_PER_UPDATE):
        state_tensor = torch.FloatTensor(np.ascontiguousarray(state)).unsqueeze(0).cuda()
        action_probs, value = model(state_tensor.cuda())

        dist = Categorical(action_probs)
        action = dist.sample()

        next_state, reward, done, _ = env.step(action.item())
        total_reward += reward

        # Store in buffer
        buffer.states.append(state)
        buffer.actions.append(action)
        buffer.log_probs.append(dist.log_prob(action).item())
        buffer.rewards.append(reward)
        buffer.dones.append(done)
        buffer.values.append(value.item())

        state = next_state
        if done:
            state = env.reset()

    # Compute advantages and returns
    advantages, returns = compute_gae(
        buffer.rewards, buffer.values, buffer.dones
    )

    # Convert buffer data to tensors
    states = torch.FloatTensor(buffer.states).cuda()
    actions = torch.tensor(buffer.actions, dtype=torch.int64).cuda()
    old_log_probs = torch.FloatTensor(buffer.log_probs).cuda()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize

    # PPO policy update
    for _ in trange(EPOCHS):
        for i in range(0, len(states), BATCH_SIZE):
            # Mini-batch
            batch_states = states[i:i + BATCH_SIZE]
            batch_actions = actions[i:i + BATCH_SIZE]
            batch_old_log_probs = old_log_probs[i:i + BATCH_SIZE]
            batch_advantages = advantages[i:i + BATCH_SIZE]
            batch_returns = returns[i:i + BATCH_SIZE]

            # Get new action probabilities and state values
            action_probs, state_values = model(batch_states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            # Ratio for clipping
            ratios = torch.exp(new_log_probs - batch_old_log_probs)

            # Surrogate objective
            surrogate1 = ratios * batch_advantages
            surrogate2 = torch.clamp(ratios, 1 - EPSILON_CLIP, 1 + EPSILON_CLIP) * batch_advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            # Value loss
            value_loss = (state_values.squeeze() - batch_returns).pow(2).mean()

            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"Episode {episode + 1}, Total Reward: {total_reward}, max value: {state.max()}")

env.close()
