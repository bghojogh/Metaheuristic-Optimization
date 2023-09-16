import random
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Callable


def local_search(cost_function: Callable, x_initial: np.array, max_itr: int, convergence_threshold: float) -> Tuple[np.array, float, List[np.array], List[float]]:
    x_current = x_initial
    cost_current = cost_function(x_current)

    x_history = [x_current]
    cost_history = [cost_current]

    # Create a tqdm progress bar
    progress_bar = tqdm(total=max_itr, desc="Iterations")

    convergence = False
    itr = 0
    while not convergence:
        # generate neighboring solutions:
        x_neighbor = [random.gauss(x, 0.1) for x in x_current]
        cost_neighbor = cost_function(x_neighbor)

        # accept the neighbor if it has lower cost:
        if cost_neighbor < cost_current:
            x_current = x_neighbor
            cost_current = cost_neighbor
            if (cost_current < convergence_threshold) or (itr >= max_itr):
                convergence = True

        x_history.append(x_current)
        cost_history.append(cost_current)

        # update the tqdm progress bar:
        progress_bar.update(1)  # Increment the progress bar by 1 unit
        itr += 1

    # get the best solution
    best_cost_index = np.argmin(cost_history)
    best_x = x_history[best_cost_index]
    best_cost = cost_history[best_cost_index]

    return best_x, best_cost, x_history, cost_history