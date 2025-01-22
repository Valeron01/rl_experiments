import cv2
import numpy as np


class SnakeGame:
    class SnakeGameActionResult:
        ACTION_PERFORMED = 0
        FOOD_EATEN = 1
        DEAD = 2
        WON = 3

    def __init__(self, field_width, field_height):
        self.field_width = field_width
        self.field_height = field_height
        self.field = np.zeros([field_height, field_width], dtype=np.uint8)
        self.snake = None
        self.score = 0

        self.up_direction = np.int32([0, -1])
        self.right_direction = np.int32([1, 0])

        self.spawn_snake()
        self.spawn_food()

    def spawn_snake(self):
        field_center_x = self.field_width // 2
        field_center_y = self.field_height // 2
        self.snake = np.int32([
            [field_center_x, field_center_y + i] for i in range(3)
        ])

        for x, y in self.snake:
            self.field[y, x] = 1
        self.field[field_center_y, field_center_x] = 2

    def spawn_food(self):
        empty_indices_y, empty_indices_x = np.nonzero(self.field == 0)
        if len(empty_indices_x) == 0:
            return False
        selected_index = np.random.randint(0, empty_indices_y.shape[0])
        self.field[empty_indices_y[selected_index], empty_indices_x[selected_index]] = 3
        return True

    def move(self, direction):
        new_head_location = self.snake[0] + direction

        if new_head_location[0] >= self.field_width or new_head_location[0] < 0:
            return SnakeGame.SnakeGameActionResult.DEAD
        if new_head_location[1] >= self.field_height or new_head_location[1] < 0:
            return SnakeGame.SnakeGameActionResult.DEAD
        value_under_head = self.field[new_head_location[1], new_head_location[0]]

        eaten_food = False
        if value_under_head == 3:
            eaten_food = True

        if value_under_head != 3 and value_under_head != 0:
            return SnakeGame.SnakeGameActionResult.DEAD

        self.field[new_head_location[1], new_head_location[0]] = 2

        self.field[self.snake[0][1], self.snake[0][0]] = 1
        self.snake = np.concatenate([new_head_location[None, :], self.snake], axis=0)

        if eaten_food:
            is_food_spawned = self.spawn_food()
            if not is_food_spawned:
                return SnakeGame.SnakeGameActionResult.WON
        else:
            self.field[self.snake[-1][1], self.snake[-1][0]] = 0
            self.snake = self.snake[:-1]

        if eaten_food:
            self.score += 1
            return SnakeGame.SnakeGameActionResult.FOOD_EATEN

        return SnakeGame.SnakeGameActionResult.ACTION_PERFORMED


    def move_up(self):
        return self.move(self.up_direction)

    def move_down(self):
        return self.move(-self.up_direction)

    def move_right(self):
        return self.move(self.right_direction)

    def move_left(self):
        return self.move(-self.right_direction)

    def reset(self):
        self.__init__(self.field_width, self.field_height)


def render_snake_field(field, snake, cell_size, padding):
    field_height, field_width = field.shape
    resulted_image = np.zeros([
        field_height * cell_size + field_height * padding + padding,
        field_width * cell_size + field_width * padding + padding,
        3
    ], dtype=np.uint8)
    colors = {
        0: [0, 0, 0],
        1: [0, 255, 0],
        2: [0, 0, 255],
        3: [255, 255, 255]
    }
    for y in range(field_height):
        for x in range(field_width):
            x_start = x * (cell_size + padding)
            y_start = y * (cell_size + padding)
            resulted_image[y_start:y_start + cell_size, x_start:x_start + cell_size] = colors[field[y, x]]

    return resulted_image


def main():
    game = SnakeGame(15, 15)
    previous_key = "w"
    for i in range(1000000):
        image = render_snake_field(game.field, game.snake, 16, 2)
        cv2.imshow("qwe", image)
        key_id = cv2.waitKey(1)
        if key_id == -1:
            key = previous_key
        else:
            key = chr(key_id)
            previous_key = key

        key = np.random.choice(["w", "a", "s", "d"])

        step_result = None
        if key == "w":
            step_result = game.move_up()
        if key == "a":
            step_result = game.move_left()
        if key == "s":
            step_result = game.move_down()
        if key == "d":
            step_result = game.move_right()

        if game.score != 0:
            print(game.score)
        if step_result == SnakeGame.SnakeGameActionResult.DEAD:
            game.reset()


if __name__ == '__main__':
    main()