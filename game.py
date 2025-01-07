import enum
import random

import cv2
import numpy as np
import numpy.random


class ActionResult:
    ACTION_BLOCKED = 0
    ACTION_PERFORMED = 1
    NO_SPACE_LEFT = 2


class Game2048:
    def __init__(self, field_size, four_spawn_probability=0.1):
        self.four_spawn_probability = four_spawn_probability
        self.field_size = field_size
        self.field = np.zeros([field_size, field_size], dtype=np.int32)
        self.spawn_number()
        self.spawn_number()

    def spawn_number(self):
        empty_y, empty_x = np.nonzero(self.field == 0)
        empty_indices = np.concatenate([empty_y[:, None], empty_x[:, None]], axis=1)
        if empty_indices.shape[0] == 0:
            return False
        selected_indices = np.random.choice(range(len(empty_indices)), [1])
        selected_indices = empty_indices[selected_indices, :]
        self.field[selected_indices[:, 0], selected_indices[:, 1]] = 2 if random.random() > self.four_spawn_probability else 4
        return True

    def move_right(self):
        action_performed = ActionResult.ACTION_BLOCKED
        merged_values = []
        field = self.field
        for i in range(self.field_size):
            last_number = None
            last_number_index = None
            for j in range(self.field_size - 1, -1, -1):
                if last_number == field[i, j]:
                    field[i, j] = 0
                    field[i, last_number_index] *= 2
                    merged_values.append(field[i, last_number_index])
                    last_number = None
                    last_number_index = None
                    action_performed = ActionResult.ACTION_PERFORMED
                    continue
                if field[i, j] != 0:
                    last_number = field[i, j]
                    last_number_index = j
                    continue
            last_empty = None
            for j in range(self.field_size - 1, -1, -1):
                if last_empty is None and field[i, j] == 0:
                    last_empty = j
                    continue
                if last_empty is None or field[i, j] == 0:
                    continue
                field[i, last_empty] = field[i, j]
                field[i, j] = 0
                last_empty = last_empty - 1
                action_performed = ActionResult.ACTION_PERFORMED

        has_space = np.count_nonzero(self.field == 0) > 0

        has_moves = False
        for i in range(self.field_size):
            if (field[i][:-1] == field[i][1:]).any() or (field[:-1, i] == field[1:, i]).any():
                has_moves = True
                break
        has_space = has_space or has_moves
        if not has_space:
            return ActionResult.NO_SPACE_LEFT, []
        if action_performed == ActionResult.ACTION_PERFORMED:
            self.spawn_number()

        return action_performed, merged_values

    def move_bottom(self):
        self.field = self.field.T
        action_performed = self.move_right()
        self.field = self.field.T
        return action_performed

    def move_top(self):
        self.field = self.field.T
        action_performed = self.move_left()
        self.field = self.field.T
        return action_performed

    def move_left(self):
        self.field = np.flip(self.field, -1)
        action_performed = self.move_right()
        self.field = np.flip(self.field, -1)
        return action_performed

    def score(self):
        return self.field.sum()


def render_field(field, cell_size, padding):
    colors = {
        2048: [230, 198, 66],
        2: [236, 228, 219],
        4: [235, 224, 203],
        8: [232, 180, 129],
        16: [223, 146, 95],
        32: [230, 131, 103],
        64: [217, 99, 67],
        0: [202, 192, 181],
    }
    scales = {
        1: 2,
        2: 2,
        3: 1.5,
        4: 1

    }
    default_color = [185, 173, 161]
    error_color = [255, 0, 255]
    field_size = field.shape[0]
    image_size = field_size * cell_size + padding * (field_size + 1)
    image = np.zeros([image_size, image_size, 3], dtype=np.uint8)
    image[:, :] = default_color
    for i in range(field_size):
        for j in range(field_size):
            cell_begin_x = padding + padding * j + cell_size * j
            cell_begin_y = padding + padding * i + cell_size * i

            image[cell_begin_y:cell_begin_y+cell_size, cell_begin_x:cell_begin_x+cell_size] = colors.get(
                field[i, j], error_color
            )

            text = str(field[i, j])
            if text == "0":
                continue
            cv2.putText(image, text, [cell_begin_x + cell_size // 3, cell_begin_y + cell_size * 2 // 3], cv2.FONT_HERSHEY_PLAIN, scales[len(text)], [0, 0, 0], 2, lineType=cv2.LINE_AA, bottomLeftOrigin=False)

    return image


def main():
    # numpy.random.seed(3)
    game = Game2048(4, 0.1)
    key_top = 2490368
    key_bottom = 2621440
    key_left = 2424832
    key_right = 2555904
    for i in range(20005):
        # print(i)
        rendered_image = render_field(game.field, 64, 8)
        cv2.imshow("2048", rendered_image[..., ::-1])
        key = cv2.waitKeyEx(0)
        print(key)
        if key == key_right:
            action_performed = game.move_right()
        elif key == key_bottom:
            action_performed = game.move_bottom()
        elif key == key_left:
            action_performed = game.move_left()
        elif key == key_top:
            action_performed = game.move_top()
        else:
            action_performed = "blyat"

        print(action_performed)



if __name__ == '__main__':
    main()