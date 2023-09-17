import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Optional


def plot_results(best_x: np.array, best_cost: float, x_history: List[np.array], cost_history: List[float], cost_function: Callable, x_range: Optional[List[List[float]]] = None) -> None:
    x1_history = [x[0] for x in x_history]
    x2_history = [x[1] for x in x_history]

    # create a 3D plot of the optimization landscape:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # generate a grid of x1 and x2 values for plotting the surface:
    if x_range is not None:
        x1_range = np.linspace(x_range[0][0], x_range[0][1], 500)
        x2_range = np.linspace(x_range[1][0], x_range[1][1], 500)
    else:
        x1_range = np.linspace(min(x1_history) - 2, max(x1_history) + 2, 500)
        x2_range = np.linspace(min(x2_history) - 2, max(x2_history) + 2, 500)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    # initialize an empty array to store the cost values:
    Z = np.zeros_like(X1)

    # calculate the cost for each combination of X1 and X2:
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i][j] = cost_function([X1[i][j], X2[i][j]])

    # plot the surface:
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

    # plot the optimization path:
    ax.plot(x1_history, x2_history, cost_history, marker='o', linestyle='-', color='red', label='Optimization path')
    ax.plot(best_x[0], best_x[1], best_cost, marker='o', linestyle='-', color='blue', label='Best solution')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Cost')
    ax.set_title('Cost function and optimization')
    plt.legend()
    plt.show()

    # calculate the extent for the 2D heatmap plot based on the actual range of the data:
    x1_min, x1_max = min(x1_range), max(x1_range)
    x2_min, x2_max = min(x2_range), max(x2_range)

    # create a 2D heatmap plot:
    plt.figure(figsize=(8, 6))
    plt.imshow(Z, extent=(x1_min, x1_max, x2_min, x2_max), origin='lower', cmap='viridis', interpolation='bilinear')
    plt.colorbar(label='Cost')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Cost function and optimization')
    plt.grid(True)

    # Overlay the optimization path on the heatmap as red dots:
    plt.plot(x1_history, x2_history, c='red', marker='o', linestyle='-', label='Optimization path')
    plt.plot(best_x[0], best_x[1], c='blue', marker='o', linestyle='-', label='Best solution')
    plt.legend()
    plt.show()