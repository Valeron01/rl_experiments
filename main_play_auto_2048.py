import cv2
import torch
import random

import torch
from torch import nn
import numpy as np
from tqdm import trange

from game import Game2048, ActionResult
from genetic_algorithm import Agent, GA2048Wrapper
from main_q_learning import QNetwork, Game2048QWrapper
from game import render_field

target_net, policy_net, optimizer, step = torch.load("./checkpoint3.pt")
# policy_net = QNetwork(1).cuda()
print("step: ", step)
assert False
game = Game2048QWrapper(4, 0.1)
# np.random.seed(0)
for i in range(20005):
    rendered_image = render_field(game.game.field, 64, 8)
    cv2.imshow("2048", rendered_image[..., ::-1])
    cv2.waitKey(10)

    inputs_tensor = torch.from_numpy(game.game.field).float()[None].cuda()
    step = policy_net.forward_epsilon_greedy(inputs_tensor, 0.00)
    print(policy_net(inputs_tensor))
    reward, done = game.make_step(step)
    print(reward)
    if reward == -0.25:
        step = policy_net.forward_epsilon_greedy(inputs_tensor, 1)
        game.make_step(step)

