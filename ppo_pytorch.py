# -*- coding: utf-8 -*-
"""
Reinforcement Learning (PPO) with TorchRL Tutorial
==================================================
**Author**: `Vincent Moens <https://github.com/vmoens>`_

This tutorial demonstrates how to use PyTorch and :py:mod:`torchrl` to train a parametric policy
network to solve the Inverted Pendulum task from the `OpenAI-Gym/Farama-Gymnasium
control library <https://github.com/Farama-Foundation/Gymnasium>`__.

.. figure:: /_static/img/invpendulum.gif
   :alt: Inverted pendulum

   Inverted pendulum

Key learnings:

- How to create an environment in TorchRL, transform its outputs, and collect data from this environment;
- How to make your classes talk to each other using :class:`~tensordict.TensorDict`;
- The basics of building your training loop with TorchRL:

  - How to compute the advantage signal for policy gradient methods;
  - How to create a stochastic policy using a probabilistic neural network;
  - How to create a dynamic replay buffer and sample from it without repetition.

We will cover six crucial components of TorchRL:

* `environments <https://pytorch.org/rl/reference/envs.html>`__
* `transforms <https://pytorch.org/rl/reference/envs.html#transforms>`__
* `models (policy and value function) <https://pytorch.org/rl/reference/modules.html>`__
* `loss modules <https://pytorch.org/rl/reference/objectives.html>`__
* `data collectors <https://pytorch.org/rl/reference/collectors.html>`__
* `replay buffers <https://pytorch.org/rl/reference/data.html#replay-buffers>`__

"""

######################################################################
# If you are running this in Google Colab, make sure you install the following dependencies:
#
# .. code-block:: bash
#
#    !pip3 install torchrl
#    !pip3 install gym[mujoco]
#    !pip3 install tqdm
#
# Proximal Policy Optimization (PPO) is a policy-gradient algorithm where a
# batch of data is being collected and directly consumed to train the policy to maximise
# the expected return given some proximality constraints. You can think of it
# as a sophisticated version of `REINFORCE <https://link.springer.com/content/pdf/10.1007/BF00992696.pdf>`_,
# the foundational policy-optimization algorithm. For more information, see the
# `Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_ paper.
#
# PPO is usually regarded as a fast and efficient method for online, on-policy
# reinforcement algorithm. TorchRL provides a loss-module that does all the work
# for you, so that you can rely on this implementation and focus on solving your
# problem rather than re-inventing the wheel every time you want to train a policy.
#
# For completeness, here is a brief overview of what the loss computes, even though
# this is taken care of by our :class:`~torchrl.objectives.ClipPPOLoss` module—the algorithm works as follows:
# 1. we will sample a batch of data by playing the
# policy in the environment for a given number of steps.
# 2. Then, we will perform a given number of optimization steps with random sub-samples of this batch using
# a clipped version of the REINFORCE loss.
# 3. The clipping will put a pessimistic bound on our loss: lower return estimates will
# be favored compared to higher ones.
# The precise formula of the loss is:
#
# .. math::
#
#     L(s,a,\theta_k,\theta) = \min\left(
#     \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), \;\;
#     g(\epsilon, A^{\pi_{\theta_k}}(s,a))
#     \right),
#
# There are two components in that loss: in the first part of the minimum operator,
# we simply compute an importance-weighted version of the REINFORCE loss (for example, a
# REINFORCE loss that we have corrected for the fact that the current policy
# configuration lags the one that was used for the data collection).
# The second part of that minimum operator is a similar loss where we have clipped
# the ratios when they exceeded or were below a given pair of thresholds.
#
# This loss ensures that whether the advantage is positive or negative, policy
# updates that would produce significant shifts from the previous configuration
# are being discouraged.
#
# This tutorial is structured as follows:
#
# 1. First, we will define a set of hyperparameters we will be using for training.
#
# 2. Next, we will focus on creating our environment, or simulator, using TorchRL's
#    wrappers and transforms.
#
# 3. Next, we will design the policy network and the value model,
#    which is indispensable to the loss function. These modules will be used
#    to configure our loss module.
#
# 4. Next, we will create the replay buffer and data loader.
#
# 5. Finally, we will run our training loop and analyze the results.
#
# Throughout this tutorial, we'll be using the :mod:`tensordict` library.
# :class:`~tensordict.TensorDict` is the lingua franca of TorchRL: it helps us abstract
# what a module reads and writes and care less about the specific data
# description and more about the algorithm itself.
#

import warnings

from game import ActionResult, Game2048

warnings.filterwarnings("ignore")
from torch import multiprocessing

# sphinx_gallery_start_ignore

# TorchRL prefers spawn method, that restricts creation of  ``~torchrl.envs.ParallelEnv`` inside
# `__main__` method call, but for the easy of reading the code switch to fork
# which is also a default spawn method in Google's Colaboratory
try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

# sphinx_gallery_end_ignore

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, IndependentNormal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

######################################################################
# Define Hyperparameters
# ----------------------
#
# We set the hyperparameters for our algorithm. Depending on the resources
# available, one may choose to execute the policy on GPU or on another
# device.
# The ``frame_skip`` will control how for how many frames is a single
# action being executed. The rest of the arguments that count frames
# must be corrected for this value (since one environment step will
# actually return ``frame_skip`` frames).
#


import numpy as np
import gym
from gym import spaces
from torchrl.envs.libs.gym import GymEnv


class Game2048PPOWrapper(gym.Env):
    def __init__(self, field_size, four_prob=0.1):
        super().__init__()
        self.game = Game2048(field_size, four_prob)
        self.field_size = field_size
        self.n_bad_steps = 0

        # Define action space (0: up, 1: right, 2: down, 3: left)
        self.action_space = spaces.Discrete(4)

        # Define observation space as the flattened game field
        self.observation_space = spaces.Box(
            low=0, high=2 ** (field_size * field_size),
            shape=(field_size, field_size), dtype=np.int32
        )

    def reset(self):
        """Resets the game environment to the initial state."""
        self.game.reset()
        self.n_bad_steps = 0
        return self._get_observation()

    def step(self, action):
        """
        Takes a step in the environment.
        Args:
            action (int): The action to take (0: up, 1: right, 2: down, 3: left).
        Returns:
            observation (np.array): The game state after the step.
            reward (float): The reward obtained from the step.
            done (bool): Whether the game has ended.
            info (dict): Additional information (optional).
        """
        if action not in range(4):
            raise ValueError(f"Invalid action: {action}")

        if action == 0:
            step_result, merged_values = self.game.move_top()
        elif action == 1:
            step_result, merged_values = self.game.move_right()
        elif action == 2:
            step_result, merged_values = self.game.move_bottom()
        elif action == 3:
            step_result, merged_values = self.game.move_left()

        reward = sum(np.power(np.log2(merged_values), 2)) if merged_values else 0
        done = False

        if step_result == ActionResult.ACTION_PERFORMED:
            self.n_bad_steps = 0
        elif step_result == ActionResult.ACTION_BLOCKED:
            reward -= 15
            self.n_bad_steps += 1
        else:
            done = True
            reward -= 500

        reward /= 50
        done = done or self.n_bad_steps == 5

        return self._get_observation(), reward, done, {}

    def render(self, mode="human"):
        """Renders the game field."""
        self.game.render()

    def _get_observation(self):
        """Returns the current state of the game."""
        return self.game.field.copy()


is_fork = multiprocessing.get_start_method() == "fork"
device = "cuda"
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames = 50_000

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4


base_env = GymEnv("InvertedDoublePendulum-v4", device=device)

env = TransformedEnv(
    base_env,
    Compose(
        # normalize observations
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter(),
    ),
)

env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

print("normalization constant shape:", env.transform[0].loc.shape)

print("observation_spec:", env.observation_spec)
print("reward_spec:", env.reward_spec)
print("input_spec:", env.input_spec)
print("action_spec (as defined by input_spec):", env.action_spec)

check_env_specs(env)

rollout = env.rollout(3)
print("rollout of three steps:", rollout)
print("Shape of the rollout TensorDict:", rollout.batch_size)

actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
)
policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
)


policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "min": env.action_spec.space.low,
        "max": env.action_spec.space.high,
    },
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)
value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device),
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)

print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)


logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for _ in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        print(tensordict_data)
        assert False
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our ``env`` horizon).
        # The ``rollout`` method of the ``env`` can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()

######################################################################
# Results
# -------
#
# Before the 1M step cap is reached, the algorithm should have reached a max
# step count of 1000 steps, which is the maximum number of steps before the
# trajectory is truncated.
#
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(logs["reward"])
plt.title("training rewards (average)")
plt.subplot(2, 2, 2)
plt.plot(logs["step_count"])
plt.title("Max step count (training)")
plt.subplot(2, 2, 3)
plt.plot(logs["eval reward (sum)"])
plt.title("Return (test)")
plt.subplot(2, 2, 4)
plt.plot(logs["eval step_count"])
plt.title("Max step count (test)")
plt.show()
