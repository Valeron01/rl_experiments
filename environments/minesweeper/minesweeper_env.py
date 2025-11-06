import math

import numpy as np

from environments.minesweeper import create_minesweeper_field, click


class MinesweeperEnv:
    def __init__(self, height, width, num_bombs, lost_reward, opened_reward, wrong_move_reward, score_multiplier):
        self.score_multiplier = score_multiplier
        self.wrong_move_reward = wrong_move_reward
        self.opened_reward = opened_reward
        self.lost_reward = lost_reward
        self.num_bombs = num_bombs
        self.width = width
        self.height = height
        self.filed, self.opened = create_minesweeper_field(height, width, num_bombs)
        self.n_steps = 0

    def score(self):
        return self.n_steps

    def state(self):
        state = self.filed / 2 - 1
        return np.stack([state * self.opened, self.opened], dtype=np.float32)

    def step(self, cell_index):
        y, x = np.unravel_index(cell_index, [self.height, self.width])
        old_opened = self.opened
        new_opened, bomb = click(self.filed, self.opened, y, x)
        self.opened = new_opened

        if bomb:
            return self.lost_reward, True

        if old_opened[y, x]:
            return self.wrong_move_reward, True

        self.n_steps += 1
        reward = self.opened_reward + self.score() * self.score_multiplier

        if old_opened[y-2:y+2, x-2:x+2].sum() == 0:
            reward = 0

        return reward, False
