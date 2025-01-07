import copy

import numpy as np
import torch

import game
from game import ActionResult
from torch import nn


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


class GAOptimizer:
    def __init__(self, model_parameters, max_mutation_rate, mutation_std):
        self.max_mutation_rate = max_mutation_rate
        self.mutation_std = mutation_std

        self.model_parameters = list(model_parameters)
        self.model_parameters_flattened = []
        elements_per_tensor = []
        for i, parameter in enumerate(self.model_parameters):
            self.model_parameters_flattened.append(parameter.view(-1))
            elements_per_tensor.append(np.prod(parameter.shape))

        self.total_elements = sum(elements_per_tensor)

        elements_per_tensor = np.float32(elements_per_tensor)
        self.probability_per_tensor = elements_per_tensor / elements_per_tensor.sum()

    def _get_random_indexes(self, indexes_count):
        selected_tensor_indexes = np.random.choice(
            range(len(self.probability_per_tensor)), size=[round(indexes_count)], p=self.probability_per_tensor
        )
        selected_item_indexes = []
        for i in selected_tensor_indexes:
            selected_item_indexes.append(np.random.randint(0, self.model_parameters[i].numel()))

        return selected_tensor_indexes, selected_item_indexes

    def mutate(self):
        elements_count = self.total_elements * np.random.random() * self.max_mutation_rate
        selected_tensors, selected_indices = self._get_random_indexes(elements_count)

        for tensor_index, element_index in zip(selected_tensors, selected_indices):
            tensor = self.model_parameters_flattened[tensor_index]
            tensor[element_index] += torch.randn([1]).to(tensor.device).item() * self.mutation_std

    def cross(self, other):
        elements_count = self.total_elements * 0.5
        selected_tensors, selected_indices = self._get_random_indexes(elements_count)

        for tensor_index, element_index in zip(selected_tensors, selected_indices):
            self.model_parameters_flattened[tensor_index][element_index] = other.model_parameters_flattened[tensor_index][element_index]


class Agent:
    def __init__(self, field_size: int, max_mutation_rate, mutation_std):
        self.network = nn.Sequential(
            nn.Linear(field_size * field_size, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 4)
        ).requires_grad_(False)

        self.optimizer = GAOptimizer(
            self.network.parameters(), max_mutation_rate=max_mutation_rate,
            mutation_std=mutation_std
        )

    def mutate(self):
        self.optimizer.mutate()
        return self

    def clone(self):
        return copy.deepcopy(self)

    def cross(self, other):
        self_copy = self.clone()
        self_copy.optimizer.cross(other)
        return self_copy

    def make_decision(self, inputs):
        inputs = torch.from_numpy(inputs).flatten()[None].float()
        inputs[inputs == 0] = 1
        inputs = torch.log2(inputs)
        inputs = inputs / 5

        inputs = inputs * 2 - 1
        res = self.network(inputs)[0]
        return res.argsort().flip(0).numpy()


def main():
    agent = Agent(4, max_mutation_rate=0.1, mutation_std=0.01)
    for i in range(100):
        inputs = np.random.uniform(size=[4, 4])
        agent.mutate()
        print(agent.make_decision(inputs))



if __name__ == '__main__':
    main()