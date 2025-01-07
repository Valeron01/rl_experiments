import numpy as np
from tqdm import trange

from genetic_algorithm import Agent, GA2048Wrapper


def main():
    total_agents = 400
    field_size = 4
    max_mutation_rate = 0.1
    mutation_std = 0.5
    total_generations = 100
    max_game_steps = 320
    stay_best_fraction = 0.2
    cross_fraction = 0.1
    agents = [Agent(
        field_size, max_mutation_rate=max_mutation_rate, mutation_std=mutation_std
    ) for _ in range(total_agents)]

    for generation in range(total_generations):
        games = [GA2048Wrapper(field_size, four_prob=0.1) for _ in agents]
        for step in trange(max_game_steps):
            for agent, game in zip(agents, games):
                step = agent.make_decision(game.state())
                game.make_step(step)

        scores_per_agent = [i.score() for i in games]

        sorting_indices = np.argsort(scores_per_agent)

        best_agents = [agents[i] for i in sorting_indices[-round(total_agents * stay_best_fraction):]]
        best_agents = best_agents + [agents[i] for i in sorting_indices[-round(total_agents * stay_best_fraction):]]
        mutated_best = [i.clone().mutate() for i in best_agents]

        cross_indices_1 = np.random.randint(0, len(best_agents), size=[round(cross_fraction * len(best_agents))])
        cross_indices_2 = np.random.randint(0, len(best_agents), size=[round(cross_fraction * len(best_agents))])

        crossed = [
            best_agents[i].cross(best_agents[i].optimizer) for i, j in zip(cross_indices_1, cross_indices_2)
        ]

        agents = best_agents + mutated_best + crossed + [Agent(
            field_size, max_mutation_rate=max_mutation_rate, mutation_std=mutation_std
        ) for _ in range(total_agents)]
        agents = agents[:total_agents]
        print(np.max(scores_per_agent), np.max([i.state().max() for i in games]), np.max([i.state().sum() for i in games]))


if __name__ == '__main__':
    main()
