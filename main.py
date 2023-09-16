from utils import search_algorithms, cost_functions, plot_utils
import yaml
from typing import Dict

def main(config: Dict) -> None:
    # get the cost function:
    if config['cost_function'] == 'sphere':
        cost_function = cost_functions.sphere
    elif config['cost_function'] == 'schwefel':
        cost_function = cost_functions.schwefel

    best_x, best_cost, x_history, cost_history = search_algorithms.local_search(cost_function=cost_function, x_initial=config['x_initial'], max_itr=config['max_itr'], convergence_threshold=config['convergence_threshold'])
    if len(config['x_initial']) == 2: 
        # if the dimensionality is 2, visualize the results.
        plot_utils.plot_results(best_x=best_x, best_cost=best_cost, x_history=x_history, cost_history=cost_history, cost_function=cost_function)

if __name__ == '__main__':
    with open('./config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config=config)