import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch

checkpoint1 = torch.load("./checkpoint3_1.pt")
checkpoint2 = torch.load("./checkpoint3_0.pt")


rewards_per_game1 = checkpoint1["max_value_per_game"]
rewards_per_game2 = checkpoint2["max_value_per_game"]


# rewards_per_game = rewards_per_game[-5000:]
# max_value_per_game = max_value_per_game[-5000:]
# epsilon_history = epsilon_history[-5000:]

rewards_per_game1 = np.clip(rewards_per_game1, -5, 10000)
rewards_per_game2 = np.clip(rewards_per_game2, -5, 10000)

# rewards_per_game = gaussian_filter1d(rewards_per_game, 2)
# max_value_per_game = gaussian_filter1d(max_value_per_game, 2)

plt.plot(rewards_per_game1)

plt.plot(rewards_per_game2)
plt.show()

fileds_sum = [np.sum(i) for i in checkpoint["fields_per_game"]]
plt.plot(fileds_sum)
plt.show()