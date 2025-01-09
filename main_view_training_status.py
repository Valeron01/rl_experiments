import matplotlib.pyplot as plt
import torch

checkpoint = torch.load("./checkpoint2_0.pt")
state_dict = checkpoint["optimizer"]()

rewards_per_game = checkpoint["rewards_per_game"]
max_value_per_game = checkpoint["max_value_per_game"]

plt.plot(rewards_per_game)
plt.show()

plt.plot(max_value_per_game)
plt.show()
