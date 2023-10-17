import yaml
from typing import Dict
from utils import search_algorithms, cost_functions


def main(config: Dict) -> None:
    cost_function = cost_functions.calculate_distances
    best_solution, best_cost = search_algorithms.aco(cost_function=cost_function, vertices=config['vertices'],
                                                     num_ants=config['num_ants'], max_itr=config['max_itr'],
                                                     max_itr_ant=config['max_itr_ant'], alpha=config['alpha'],
                                                     beta=config['beta'], pheromone_decay=config['pheromone_decay'],
                                                     starting_vertex=config['starting_vertex'], stopping_vertex=config['stopping_vertex'],
                                                     one_time_visit=config['one_time_visit'], visit_all_vertices=config['visit_all_vertices'],
                                                     TSP=config['TSP'])

    print(f"Best Solution: {best_solution}")
    print(f"Best Cost: {best_cost}")

if __name__ == '__main__':
    with open('./config/config_ACO.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config=config)