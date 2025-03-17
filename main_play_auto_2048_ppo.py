import cv2
import random

import torch

from game import render_field
from main_ppo_2048 import Game2048PPOWrapper, PPOTransformerNetwork

# 32 28 35
model = torch.load("/home/valera/PycharmProjects/TwentyFourtyEight/logs_ppo_2048/run_35/Checkpoints/CheckpointBackup.pt")
model = model.eval().requires_grad_(False)
# policy_net = QNetwork(1).cuda()

# assert False

game = Game2048PPOWrapper(4, 0.1)
# np.random.seed(0)
for i in range(20005):
    rendered_image = render_field(game.game.field, 64, 8)
    inputs_tensor = torch.from_numpy(game.game.field).float()[None].cuda()
    actor, value = model(inputs_tensor)
    probs = actor.probs.cpu()[0]
    step = probs.argmax().item()
    reward, done = game.make_step(step)
    if reward == -0.5:
        step = random.randrange(0, 4)
        reward, done = game.make_step(step)
    print(reward, probs, step, value.item())
    cv2.imshow("2048", rendered_image[..., ::-1])
    cv2.waitKey(1)

