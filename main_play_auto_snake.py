import cv2
import numpy as np
import torch

from main_q_learning_snake import QNetwork
from snake_game import SnakeGame, render_snake_field


def main():
    game = SnakeGame(16, 16)
    checkpoint = torch.load("checkpoints_dqn_snake/checkpoint2_1.pt")

    policy_net = QNetwork(**checkpoint["model_parameters"]).cuda().eval()
    policy_net.load_state_dict(checkpoint["policy_net"])  # target_net|policy_net

    for i in range(10000):
        image = render_snake_field(game.field, game.snake, 16, 2)
        cv2.imshow("qwe", image)
        key_id = cv2.waitKey(1)

        inputs_tensor = torch.from_numpy(game.field).float()[None].cuda()
        step = policy_net.forward_epsilon_greedy(inputs_tensor, 0.00)

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