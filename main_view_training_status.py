import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch

checkpoint = torch.load("./checkpoint2_2.pt")
state_dict = checkpoint["optimizer"]

rewards_per_game = checkpoint["rewards_per_game"]
max_value_per_game = checkpoint["max_value_per_game"]
epsilon_history = checkpoint["epsilon_history"]

# rewards_per_game = rewards_per_game[-5000:]
# max_value_per_game = max_value_per_game[-5000:]
# epsilon_history = epsilon_history[-5000:]

rewards_per_game = np.clip(rewards_per_game, -5, 10000)

# rewards_per_game = gaussian_filter1d(rewards_per_game, 2)
# max_value_per_game = gaussian_filter1d(max_value_per_game, 2)

plt.plot(rewards_per_game)
plt.show()

plt.plot(max_value_per_game)
plt.show()

plt.plot(epsilon_history)
plt.show()

fileds_sum = [np.sum(i) for i in checkpoint["fields_per_game"]]
plt.plot(fileds_sum)
plt.show()