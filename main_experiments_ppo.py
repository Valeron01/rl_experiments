import torch

from main_ppo_my import SnakePPOWrapper, PPONetwork

loaded_model = torch.load("/home/valera/PycharmProjects/TwentyFourtyEight/checkpoints_ppo_snake_my/checkpoint_0_0.pt")
env = SnakePPOWrapper(16)

state = torch.from_numpy(env.game.field).cuda()[None]

distribution, returns = loaded_model(state)
print(distribution.probs)
print(returns)