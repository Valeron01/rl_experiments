import torch

from main_ppo_my import SnakePPOWrapper, PPONetwork
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
    loaded_model = torch.load("/home/valera/PycharmProjects/TwentyFourtyEight/checkpoints_ppo_snake_my/checkpoint_2_0.pt").eval().requires_grad_(False)

    for i in range(10000):
        inputs_tensor = torch.from_numpy(game.field).float()[None].cuda()
        step_dist, value = loaded_model(inputs_tensor)
        probs = step_dist.probs[0].cpu().numpy()
        step = probs.argmax().item()
        print("--------------------")
        for j in range(4):
            print(f"{index_to_string(j)}: {probs[j]:.4f}")
        print(f"Total: {index_to_string(step)}")


        image = render_snake_field(game.field, game.snake, 16, 2)
        cv2.imshow("qwe", image)
        key_id = cv2.waitKey(0)

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
            print(game.score)
        if step_result == SnakeGame.SnakeGameActionResult.DEAD:
            game.reset()


if __name__ == '__main__':
    main()