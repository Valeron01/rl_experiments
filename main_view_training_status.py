import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch

checkpoint = torch.load("/home/valera/PycharmProjects/TwentyFourtyEight/checkpoints_dqn_snake/checkpoint2_1.pt")
# print(checkpoint["epoch"])
print(checkpoint["hyper_parameters"])
print(checkpoint["model_parameters"])
# assert False

rewards_per_game = checkpoint["rewards_per_game"]
max_value_per_game = checkpoint["max_value_per_game"]
epsilon_history = checkpoint["epsilon_history"]

rewards_per_game = gaussian_filter1d(rewards_per_game, 3)

plt.plot(rewards_per_game)
plt.show()

plt.plot(max_value_per_game)
plt.show()

fileds_sum = [np.sum(i) for i in checkpoint["fields_per_game"]]
plt.plot(fileds_sum)
plt.show()