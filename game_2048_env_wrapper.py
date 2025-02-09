import game
from game import ActionResult


class GA2048Wrapper:
    def __init__(self, field_size, four_prob=0.1):
        self.game = game.Game2048(field_size, four_prob)
        self.steps_history = []

    def make_step(self, step_index):
        step_index = step_index[0]
        if step_index == 0:
            step_result = self.game.move_top()
        elif step_index == 1:
            step_result = self.game.move_right()
        elif step_index == 2:
            step_result = self.game.move_bottom()
        elif step_index == 3:
            step_result = self.game.move_left()
        else:
            raise NotImplementedError()
        self.steps_history.append(step_result)

    def score(self):
        steps_score = 0
        for i in self.steps_history:
            if i == ActionResult.ACTION_PERFORMED:
                steps_score += 6
            elif i == ActionResult.ACTION_BLOCKED:
                steps_score -= 15
                break
            elif i == ActionResult.NO_SPACE_LEFT:
                steps_score -= 10
                break

        return self.game.field.max() * 2 + self.game.score() + steps_score

    def state(self):
        return self.game.field

