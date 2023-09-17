from utils import search_algorithms, cost_functions, plot_utils
import yaml
from typing import Dict

def main(config: Dict) -> None:
    # get the cost function:
    if config['cost_function'] == 'sphere':
        cost_function = cost_functions.sphere
        x_range = [[-100, 100], [-100, 100]]  # the range for each dimension
    elif config['cost_function'] == 'schwefel':
        cost_function = cost_functions.schwefel
        x_range = [[-500, 500], [-500, 500]]  # the range for each dimension

    best_x, best_cost, x_history, cost_history = search_algorithms.local_search(cost_function=cost_function, max_itr=config['max_itr'], convergence_threshold=config['convergence_threshold'], 
                                                                                x_initial=config['x_initial'], x_range=x_range)
    if len(best_x) == 2: 
        # if the dimensionality is 2, visualize the results.
        plot_utils.plot_results(best_x=best_x, best_cost=best_cost, x_history=x_history, cost_history=cost_history, cost_function=cost_function, x_range=x_range)

if __name__ == '__main__':
    with open('./config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config=config)