import random
from collections import deque

import torch
from torch import nn
import numpy as np
from tqdm import trange

from game import Game2048, ActionResult
from genetic_algorithm import Agent, GA2048Wrapper


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

        resulted_distribution = torch.distributions.Categorical(actor_head)

        return resulted_distribution, critic_head


class Game2048PPOWrapper:
    def __init__(self, field_size, four_prob=0.1):
        self.game = Game2048(field_size, four_prob)
        self.n_bad_steps = 0

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
            self.n_bad_steps = 0
        elif step_result == ActionResult.ACTION_BLOCKED:
            reward += -15
            self.n_bad_steps += 1
        else:
            done = True
            reward += -500

        reward /= 50

        done = done or self.n_bad_steps == 5

        return reward, done


def compute_gae(next_value, rewards, masks, values, gamma=0.999, tau=0.95):
    # Similar to calculating the returns we can start at the end of the sequence and go backwards
    gae = 0
    returns = deque()
    gae_logger = deque()

    for step in reversed(range(len(rewards))):
        # Calculate the current delta value
        delta = rewards[step] + gamma * next_value * masks[step] - values[step]

        # The GAE is the decaying sum of these delta values
        gae = delta + gamma * tau * masks[step] * gae

        # Get the new next value
        next_value = values[step]

        # If we add the value back to the GAE we get a TD approximation for the returns
        # which we can use to train the Value function
        returns.appendleft(gae + values[step])
        gae_logger.appendleft(gae)

    return returns, gae_logger


def ppo_loss(new_dist, actions, old_log_probs, advantages, clip_param):
    ########### Policy Gradient update for actor with clipping - PPO #############

    # 1. Find the new probability
    # Work out the probability (log probability) that the agent will NOW take
    # the action it took during the rollout
    # We assume there has been some optimisation steps between when the action was taken and now so the
    # probability has probably changed
    new_log_probs = new_dist.log_prob(actions)

    # 2. Find the ratio of new to old - r_t(theta)
    # Calculate the ratio of new/old action probability (remember we have log probabilities here)
    # log(new_prob) - log(old_prob) = log(new_prob/old_prob)
    # exp(log(new_prob/old_prob)) = new_prob/old_prob
    # We use the ratio of new/old action probabilities (not just the log probability of the action like in
    # vanilla policy gradients) so that if there is a large difference between the probabilities then we can
    # take a larger/smaller update step
    # EG: If we want to decrease the probability of taking an action but the new action probability
    # is now higher than it was before we can take a larger update step to correct this
    ratio = (new_log_probs - old_log_probs).exp()

    # 3. Calculate the ratio * advantage - the first term in the MIN statement
    # We want to MAXIMISE the (Advantage * Ratio)
    # If the advantage is positive this corresponds to INCREASING the probability of taking that action
    # If the advantage is negative this corresponds to DECREASING the probability of taking that action
    surr1 = ratio * advantages

    # 4. Calculate the (clipped ratio) * advantage - the second term in the MIN statement
    # PPO goes a bit further, if we simply update update using the Advantage * Ratio we will sometimes
    # get very large or very small policy updates when we don't want them
    #
    # EG1: If we want to increase the probability of taking an action but the new action probability
    # is now higher than it was before we will take a larger step, however if the action probability is
    # already higher we don't need to keep increasing it (large output values can create instabilities).
    #
    # EG2: You can also consider the opposite case where we want to decrease the action probability
    # but the probability has already decreased, in this case we will take a smaller step than before,
    # which is also not desirable as it will slow down the "removal" (decreasing the probability)
    # of "bad" actions from our policy.
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages

    # 5. Take the minimum of the two "surrogate" losses
    # PPO therefore clips the upper bound of the ratio when the advantage is positive
    # and clips the lower bound of the ratio when the advantage is negative so our steps are not too large
    # or too small when necessary, it does this by using a neat trick of simply taking the MIN of two "surrogate"
    # losses which chooses which loss to use!
    actor_loss = torch.min(surr1, surr2)

    # 6. Return the Expectation over the batch
    return actor_loss.mean()


def clipped_critic_loss(new_value, old_value, returns, clip_param):
    ########### Value Function update for critic with clipping #############

    # To help stabalise the training of the value function we can do a similar thing as the clipped objective
    # for PPO - Note: this is NOT nessisary but does help!

    # 1. MSE/L2 loss on the current value and the returns
    vf_loss1 = (new_value - returns).pow(2.)

    # 2. MSE/L2 loss on the clipped value and the returns
    # Here we create an "approximation" of the new value (aka the current value) by finding the difference
    # between the "new" and "old" value and adding a clipped amount back to the old value
    vpredclipped = old_value + torch.clamp(new_value - old_value, -clip_param, clip_param)
    # Note that we ONLY backprop through the new value
    vf_loss2 = (vpredclipped - returns).pow(2.)

    # 3. Take the MAX between the two losses
    # This trick has the effect of only updating the current value DIRECTLY if is it WORSE (higher error)
    # than the old value.
    # If the old value was worse then the "approximation" will be worse and we update
    # the new value only a little bit!
    critic_loss = torch.max(vf_loss1, vf_loss2)

    # 4. Return the Expectation over the batch
    return critic_loss.mean()


def ppo_update(model, optimizer, states, actions, log_probs, advantages, values, returns, ppo_epochs, clip_param, batch_size=128):
    n_batches = states.shape[0] // batch_size
    n_batches = max(1, n_batches)

    batched_indices = torch.randperm(states.shape[0])
    batched_indices = torch.tensor_split(batched_indices, n_batches)

    for _ in range(ppo_epochs):
        for batch_indices in batched_indices:
            batch_multiplier = batch_indices.shape[0] / batch_size
            # Forward pass of input state observationsequence
            new_dist, new_value = model(states[batch_indices])

            # Most Policy gradient algorithms include a small "Entropy bonus" to increases the "entropy" of
            # the action distribution, aka the "randomness"
            # This ensures that the actor does not converge to taking the same action everytime and
            # maintains some ability for "exploration" of the policy

            # Determine expectation over the batch of the action distribution entropy
            entropy = new_dist.entropy().mean()

            actor_loss = ppo_loss(
                new_dist, actions[batch_indices], log_probs[batch_indices], advantages[batch_indices], clip_param
            )
            critic_loss = clipped_critic_loss(new_value, values[batch_indices], returns[batch_indices], clip_param)


            # These techniques allow us to do multiple epochs of our data without huge update steps throwing off our
            # policy/value function (gradient explosion etc).
            # It can also help prevent "over-fitting" to a single batch of observations etc,
            # RL boot-straps itself and the noisy "ground truth" targets (if you can call them that) will
            # shift overtime and we need to make sure our actor-critic can quickly adapt, over-fitting to a
            # single batch of observations will prevent that
            agent_loss = critic_loss - actor_loss - 0.01 * entropy
            agent_loss = agent_loss * batch_multiplier

            optimizer.zero_grad()
            agent_loss.backward()
            # Clip gradient norm to further prevent large updates
            nn.utils.clip_grad_norm_(model.parameters(), 40)
            optimizer.step()



def main():
    num_game_steps = 4096
    field_size = 4
    num_epochs = 10_000_000
    batch_size = 128
    lr = 1e-4
    weight_decay = 1e-2
    gamma = 0.7
    tau = 0.95
    clip_param = 0.2
    random_epsilon_start = 35_000
    start_epoch = 0
    ppo_epochs = 1

    model_parameters = {
        "field_size": field_size,
        "d_model": 256,
        "n_heads": 4,
        "dim_feedforward": 512,
        "n_layers": 5
    }

    policy_net = PPONetwork(**model_parameters).cuda()
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

    max_value_per_game = []
    rewards_per_game = []
    fields_per_game = []

    # if False:
    #     loaded_checkpoint = torch.load("./checkpoint3_5.pt")
    #     optimizer.load_state_dict(loaded_checkpoint["optimizer"])
    #     policy_net.load_state_dict(loaded_checkpoint["policy_net"])
    #     start_epoch = loaded_checkpoint["epoch"] + 1
    #     max_value_per_game = loaded_checkpoint["max_value_per_game"]
    #     rewards_per_game = loaded_checkpoint["rewards_per_game"]
    #
    #     fields_per_game = loaded_checkpoint["fields_per_game"]

    for epoch in range(start_epoch, num_epochs):

        game = Game2048PPOWrapper(field_size)

        states = []
        log_probs = []
        rewards = []
        next_states = []
        dones = []
        actions = []
        values = []

        game_reward = 0
        policy_net = policy_net.eval()
        for game_iter in range(num_game_steps):
            game_state = torch.from_numpy(game.game.field.copy())[None]
            with torch.inference_mode():
                dist, value = policy_net(game_state.cuda())
            values.append(value)
            action = dist.sample()
            actions.append(action)
            reward, done = game.make_step(action.item())
            next_game_state = torch.from_numpy(game.game.field.copy())[None]

            game_reward += reward

            states.append(game_state)
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            next_states.append(next_game_state)
            dones.append(done)

            if done:
                break

        states = torch.cat(states, dim=0).cuda()
        log_probs = torch.cat(log_probs, dim=0).cuda()
        dones = torch.FloatTensor(dones).cuda()
        actions = torch.cat(actions, 0)
        values = torch.cat(values, 0)

        with torch.inference_mode():
            _, next_value = policy_net(next_game_state.cuda())

        returns, advantage = compute_gae(next_value, rewards, 1 - dones, values, gamma=gamma, tau=tau)
        returns = list(returns)
        advantage = list(advantage)

        returns = torch.cat(returns, 0)
        advantage = torch.cat(advantage, 0)

        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)


        policy_net.train()
        ppo_update(
            policy_net,
            optimizer,
            states,
            actions,
            log_probs,
            advantage,
            values,
            returns,
            ppo_epochs=ppo_epochs,
            clip_param=clip_param,
            batch_size=batch_size
        )

        max_value_per_game.append(game.game.field.max())
        rewards_per_game.append(game_reward)
        fields_per_game.append(game.game.field)
        print(game.game.field.max(), game_reward, epoch)
        if epoch % 1000 == 0:
            print("Saving checkpoint")
            torch.save({
                "policy_net": policy_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "model_parameters": model_parameters,
                "hyper_parameters": {
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
                "rewards_per_game": rewards_per_game,
                "fields_per_game": fields_per_game
            }, "./checkpoints_ppo/0_0.pt")


if __name__ == '__main__':
    main()
