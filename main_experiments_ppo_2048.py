import numpy as np
import torch

from main_ppo_my import SnakePPOWrapper, PPONetwork, PPOResidualNetwork, ResBlock, PPOResidualNetwork2, PPOResidualNetwork3
import cv2
import torch

from main_q_learning_snake import QNetwork
from snake_game import SnakeGame, render_snake_field


def index_to_string(step):
    if step == 0:
        return "up"
    if step == 3:
        return "le"
    if step == 2:
        return "do"
    if step == 1:
        return "ri"

def main():
    game = SnakeGame(16, 16)
    # loaded_model = torch.load("/home/valera/PycharmProjects/TwentyFourtyEight/checkpoints_ppo_snake_my/checkpoint_2_0.pt").eval().requires_grad_(False)
    loaded_model = torch.load("/logs_ppo_snake/outdated/run_117/Checkpoints/Checkpoint.pt").eval().requires_grad_(False)
    scores = []
    for i in range(100000):
        inputs_tensor = torch.from_numpy(game.field).float()[None].cuda()
        step_dist, value = loaded_model(inputs_tensor)
        probs = step_dist.probs[0].cpu().numpy()
        step = probs.argmax().item()



        # image = render_snake_field(game.field, game.snake, 16, 2)
        image = cv2.resize(game.field, [256, 256], interpolation=cv2.INTER_NEAREST).astype(np.float32)
        image = image - image.min()
        image = image / image.max()
        cv2.imshow("qwe", image)
        key_id = cv2.waitKey(50)

        step_result = None
        if step == 0:
            step_result = game.move_up()
        if step == 3:
            step_result = game.move_left()
        if step == 2:
            step_result = game.move_down()
        if step == 1:
            step_result = game.move_right()

        if game.score != 0:
            pass
        if step_result == SnakeGame.SnakeGameActionResult.DEAD:
            scores.append(game.score)
            print(max(scores))
            print("--------------------")
            for j in range(4):
                print(f"{index_to_string(j)}: {probs[j]:.4f}")
            print(f"Total: {index_to_string(step)}")
            cv2.waitKey(0)
            game.reset()


if __name__ == '__main__':
    main()