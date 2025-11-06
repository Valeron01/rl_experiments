import cv2
import numpy as np
import torch
import tqdm

from environments.minesweeper.minesweeper_env import MinesweeperEnv
from environments.minesweeper import draw_minesweeper

from main_train_minesweeper import MinesweeperConvModel


def main():
    device = "cuda:0"
    env_params = {
        "height": 16,
        "width": 32,
        "num_bombs": 80,
        "lost_reward": -0.01,
        "opened_reward": 0.005,
        "wrong_move_reward": -0.001,
        "score_multiplier": 0.003
    }
    model = MinesweeperConvModel().to(device).eval().requires_grad_(False)
    checkpoint = torch.load("./logs_reinforce_minesweeper/run_066/Checkpoints/Checkpoint.pt")
    print(checkpoint["optimizer"])
    model.load_state_dict(
        checkpoint["state_dict"]
    )
    environment = MinesweeperEnv(**env_params)

    done = False
    printer = tqdm.trange(5000)
    scores = []
    for i in printer:
        rendered_image = draw_minesweeper(
            environment.filed, environment.opened, cell_size=25
        )
        cv2.imshow("NNPlaying", rendered_image)
        cv2.waitKey(0)

        if done:
            scores.append(environment.score())
            printer.set_postfix({
                "mean": np.mean(scores)
            })
            environment = MinesweeperEnv(**env_params)
        state = environment.state()
        prediction = model(torch.from_numpy(state[None]).to(device))
        probs = prediction.logits.softmax(-1).view(environment.height, environment.width).cpu()
        probs = probs - probs.min()
        cv2.imshow("Probs", cv2.resize(probs.numpy(), None, fx=16, fy=16, interpolation=cv2.INTER_NEAREST))
        action = prediction.sample().item()
        reward, done = environment.step(action)


if __name__ == '__main__':
    main()
