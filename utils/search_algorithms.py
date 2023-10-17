"""
This is an implementation of three metaheuristic optimization algorithms.
    1. Local Search (Hill Climbing)
    2. Iterative Local Search
    3. Simulated Annealing
"""

import random
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Callable, Optional
import matplotlib.pyplot as plt


##############################################################################################################
############ Local Search (Hill Climbing) Algorithm ##########################################################
##############################################################################################################
def local_search(cost_function: Callable, max_itr: int, convergence_threshold: float, 
                 x_initial: Optional[np.array] = None, x_range: Optional[List[List[float]]] = None, hide_progress_bar: Optional[bool] = False) -> Tuple[np.array, float, List[np.array], List[float]]:
    # Set the x_initial
    if x_initial is None:
        x_initial = [random.uniform(x_range[i][0], x_range[i][1]) for i in range(len(x_range))]

    x_current = x_initial
    cost_current = cost_function(x_current)

    x_history = [x_current]
    cost_history = [cost_current]

    # Create a tqdm progress bar
    if not hide_progress_bar:
        progress_bar = tqdm(total=max_itr, desc='Iterations')

    convergence = False
    itr = 0
    while not convergence:
        # Generate neighboring solutions
        x_neighbor = [random.gauss(x, 0.1) for x in x_current]
        x_neighbor = bound_solution_in_x_range(x=x_neighbor, x_range=x_range)
        cost_neighbor = cost_function(x_neighbor)

        # Accept the neighbor if it has lower cost
        if cost_neighbor < cost_current:
            x_current = x_neighbor
            cost_current = cost_neighbor
            if (cost_current < convergence_threshold) or (itr >= max_itr):
                convergence = True

        x_history.append(x_current)
        cost_history.append(cost_current)

        # Update the tqdm progress bar
        if not hide_progress_bar:
            progress_bar.update(1)  # Increment the progress bar by 1 unit
        itr += 1
    
    # progress_bar.close()

    # Get the best solution
    best_cost_index = np.argmin(cost_history)
    best_x = x_history[best_cost_index]
    best_cost = cost_history[best_cost_index]

    return best_x, best_cost, x_history, cost_history

##############################################################################################################
############ Iterative Local Search Algorithm ################################################################
##############################################################################################################
def iterative_local_search(cost_function: Callable, max_itr_ils: int, max_itr_ls: int, convergence_threshold: float,
                           x_initial: Optional[np.array] = None, x_range: Optional[List[List[float]]] = None) -> Tuple[np.array, float, List[np.array], List[float]]:
    # Set the x_initial
    if x_initial is None:
        x_initial = [random.uniform(x_range[i][0], x_range[i][1]) for i in range(len(x_range))]

    x_current = x_initial
    cost_current = cost_function(x_current)

    x_history = [x_current]
    cost_history = [cost_current]

    # Create a tqdm progress bar
    progress_bar = tqdm(total=max_itr_ils, desc='Iterations')
    
    for _ in range(max_itr_ils):
        # Do local search
        best_x, best_cost, _, _ = local_search(cost_function=cost_function, max_itr=max_itr_ls, convergence_threshold=convergence_threshold,
                                               x_initial=x_current, x_range=x_range)
        x_history.append(best_x)
        cost_history.append(best_cost)
        
        # Sample from the optimization landscape
        x_current = [random.uniform(x_range[i][0], x_range[i][1]) for i in range(len(x_range))]

        # Update the tqdm progress bar
        progress_bar.update(1)  # Increment the progress bar by 1 unit

    progress_bar.close()
    
    # Get the best solution
    best_cost_index = np.argmin(cost_history)
    best_x = x_history[best_cost_index]
    best_cost = cost_history[best_cost_index]

    return best_x, best_cost, x_history, cost_history

##############################################################################################################
############ Simulated Annealing #############################################################################
##############################################################################################################
def simulated_annealing(cost_function: Callable, max_itr: int, temperature: float, alpha: float, beta: float,
                        x_initial: Optional[np.array] = None, x_range: Optional[List[List[float]]] = None,
                        temperature_decrement_method: Optional[str] = 'linear') -> Tuple[np.array, float, List[np.array], List[float]]:
    # Set the x_initial
    if x_initial is None:
        x_initial = [random.uniform(x_range[i][0], x_range[i][1]) for i in range(len(x_range))]

    x_current = x_initial
    cost_current = cost_function(x_current)

    x_history = [x_current]
    cost_history = [cost_current]

    # Set the initial temperature
    T = temperature

    # Create a tqdm progress bar
    progress_bar = tqdm(total=max_itr, desc='Iterations')

    itr = 0
    while (itr <= max_itr):
        # Generate neighboring candidates
        x_neighbor = [random.gauss(x, 0.1) for x in x_current]
        x_neighbor = bound_solution_in_x_range(x=x_neighbor, x_range=x_range)
        cost_neighbor = cost_function(x_neighbor)

        # Calculate ∆E
        Delta_E = cost_neighbor - cost_current

        # Accept the neighbor if it has lower cost
        if Delta_E <= 0:
            x_current = x_neighbor
            cost_current = cost_neighbor
            x_history.append(x_current)
            cost_history.append(cost_current)
        else:
            u = random.uniform(0, 1)
            if (u <= np.exp(-Delta_E / T)):
                x_current = x_neighbor
                cost_current = cost_neighbor
                x_history.append(x_current)
                cost_history.append(cost_current)
        
        # Decrement the temperature T
        if temperature_decrement_method == 'linear':
            T = T - alpha  # Linear reduction rule
        elif temperature_decrement_method == 'geometric':
            T = T * alpha  # Geometric reduction rule
        elif temperature_decrement_method == 'slow':
            T = T / (1 + (beta * T))  # Slow-decrease rule

        # Update the tqdm progress bar
        progress_bar.update(1)  # Increment the progress bar by 1 unit
        itr += 1
    
    progress_bar.close()

    # Get the best solution
    best_cost_index = np.argmin(cost_history)
    best_x = x_history[best_cost_index]
    best_cost = cost_history[best_cost_index]

    return best_x, best_cost, x_history, cost_history

##############################################################################################################
############ Particle Swarm Optimization (PSO) ###############################################################
##############################################################################################################
def pso(cost_function: Callable, num_particles: int, max_itr: int, alpha_1: float, alpha_2: float, alpha_3: float,
        x_initial: Optional[np.array] = None, x_range: Optional[List[List[float]]] = None,
        local_best_option: Optional[str] = 'this_iteration', global_best_option: Optional[str] = 'this_iteration',
        ls_max_itr: Optional[int] = 100, ls_convergence_threshold: Optional[float] = 0.01) -> Tuple[np.array, float, List[np.array], List[float]]:
    # Set the x_initial
    if x_initial is None:
        x_initial = [random.uniform(x_range[i][0], x_range[i][1]) for i in range(len(x_range))]
    
    # Initialize particles (candidate solutions)
    particles = [{'position': np.array([random.uniform(x[0], x[1]) for x in x_range]),
                  'velocity': np.array([random.uniform(-1, 1) for _ in range(len(x_range))]),
                  'best_position': x_initial,
                  'best_cost': float('inf'),
                  'position_history': []
                  } for _ in range(num_particles)]

    # Initialize global best
    global_best_position = x_initial
    global_best_cost = float('inf')
    
    x_history = []
    cost_history = []

    progress_bar = tqdm(total=max_itr, desc='Iterations')

    for _ in range(max_itr):
        best_xs_in_this_iteration, best_costs_in_this_iteration = [], []
        
        for particle_index in range(len(particles)):
            # Do local search (every particle searches locally in the local neighborhood)
            best_x, best_cost, _, _ = local_search(cost_function=cost_function, max_itr=ls_max_itr, convergence_threshold=ls_convergence_threshold,
                                                   x_initial=particles[particle_index]['position'], x_range=x_range, hide_progress_bar=True)

            # Find the local best particle (for use in the velocity vector):
            if local_best_option == 'this_iteration':
                local_best_x = best_x
            elif local_best_option == 'so_far':
                if best_cost < particles[particle_index]['best_cost']:
                    particles[particle_index]['best_cost'] = best_cost
                    particles[particle_index]['best_position'] = best_x
                    local_best_x = particles[particle_index]['best_position']
            
            best_xs_in_this_iteration.append(best_x)
            best_costs_in_this_iteration.append(best_cost)

        # Find the best solution of this iteration
        best_cost_index_in_this_iteration = np.argmin(best_costs_in_this_iteration)
        best_cost_in_this_iteration = best_costs_in_this_iteration[best_cost_index_in_this_iteration]
        best_x_in_this_iteration = best_xs_in_this_iteration[best_cost_index_in_this_iteration]
        if best_cost_in_this_iteration < global_best_cost:
            global_best_cost = best_cost_in_this_iteration
            global_best_position = best_x_in_this_iteration
        
        # Find the global best particle (for use in the velocity vector):
        if global_best_option == 'this_iteration':
            global_best_x = best_x_in_this_iteration
        elif global_best_option == 'so_far':
            global_best_x = global_best_position

        # Update every particle (using regularization hyper-parameters)
        for particle_index in range(len(particles)):
            particles[particle_index]['velocity'] = (alpha_1 * particles[particle_index]['velocity'] +
                                                     alpha_2 * (local_best_x - particles[particle_index]['position']) +
                                                     alpha_3 * (global_best_x - particles[particle_index]['position']))
            particles[particle_index]['position'] = particles[particle_index]['position'] + particles[particle_index]['velocity']
            particles[particle_index]['position'] = bound_solution_in_x_range(x=particles[particle_index]['position'], x_range=x_range)
            particles[particle_index]['position_history'].append(particles[particle_index]['position'])

        x_history.append(global_best_position)
        cost_history.append(global_best_cost)

        # Update the tqdm progress bar
        progress_bar.update(1)  # Increment the progress bar by 1 unit

    progress_bar.close()

    return global_best_position, global_best_cost, x_history, cost_history, particles

##############################################################################################################
############ Genetic Algorithm (GA) ##########################################################################
##############################################################################################################
def ga(cost_function: Callable, population_size: int, max_itr: int, mutation_rate: float, crossover_rate: float,
       x_initial: Optional[np.array] = None, x_range: Optional[List[List[float]]] = None) -> Tuple[np.array, float, List[np.array], List[float]]:
    # Initialize the population
    population = [np.array([random.uniform(r[0], r[1]) for r in x_range]) for _ in range(population_size)]

    # Create a tqdm progress bar
    progress_bar = tqdm(total=max_itr, desc="Generations")

    best_solution = x_initial
    best_cost = float('inf')
    history = {'best_costs': [], 'best_solutions': []}

    # Initialize chromosome history (required for visualization)
    chromosomes = [{
                    'position_history': []
                   } for _ in range(population_size)]

    for _ in range(max_itr):
        # Evaluate the cost of each individual in the population
        cost_values = [cost_function(individual) for individual in population]  # individuals = candidate solutions

        # Update the chromosome history (required for visualization)
        for chromosome_index in range(len(chromosomes)):
            chromosomes[chromosome_index]['position_history'].append(population[chromosome_index])

        # Find the best solution in this generation/iteration
        best_generation_cost = min(cost_values)
        best_generation_index = cost_values.index(best_generation_cost)
        best_generation_solution = population[best_generation_index]
        if best_generation_cost < best_cost:
            best_solution = best_generation_solution
            best_cost = best_generation_cost
        history['best_costs'].append(best_cost)
        history['best_solutions'].append(best_solution)

        # Select parents for crossover (natural selection)
        num_parents = int(population_size * crossover_rate)  # Number of parents is selected as a fraction of population size
        parents_indices = np.argsort(cost_values)[:num_parents]  # Select the first num_parents indices corresponding to the individuals with lowest cost_values
        parents = [population[i] for i in parents_indices]

        # Create offspring through crossover
        offspring = []
        while len(offspring) < population_size:

            # Random natural selection
            parent1, parent2 = random.sample(parents, k=2)

            # One-point crossover
            crossover_point = random.randint(1, len(parent1) - 1)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            offspring.append(child)

        # Mutate offspring (random changes to the offsprings after crossover)
        for i in range(len(offspring)):
            if random.uniform(0, 1) < mutation_rate:
                mutation_point = random.randint(0, len(x_range) - 1)
                offspring[i][mutation_point] = random.uniform(x_range[mutation_point][0], x_range[mutation_point][1])

        # Replace the old population with the new population (offspring)
        population = offspring

        # Update the tqdm progress bar
        progress_bar.update(1)  # Increment the progress bar by 1 unit

    progress_bar.close()

    return best_solution, best_cost, history['best_solutions'], history['best_costs'], chromosomes

##############################################################################################################
############ Ant Colony Optimization (ACO) ###################################################################
##############################################################################################################
def aco(cost_function: Callable, vertices: List[List[float]], num_ants: int, max_itr: int, max_itr_ant: int, alpha: float, beta: float,
        pheromone_decay: float, starting_vertex: Optional[int] = None, stopping_vertex: Optional[int] = None,
        one_time_visit: Optional[bool] = False, visit_all_vertices: Optional[bool] = False,
        TSP: Optional[bool] = False,) -> Tuple[np.array, float, List[np.array], List[float]]:
    
    graph = cost_function(vertices)
    num_vertices = len(vertices)
    
    # Initialize the pheromones of all edges to equal amount (small positive random values at the start of ACO algorithm)
    # pheromone = [[0.5] * num_vertices for _ in range(num_vertices)]
    pheromone = [[np.random.uniform(0, 1)] * num_vertices for _ in range(num_vertices)]
    
    best_solution = []
    best_cost = float('inf')
    
    # Create a tqdm progress bar
    progress_bar = tqdm(total=max_itr, desc='Iterations')

    itr = 0
    while (itr <= max_itr):
        solutions = []
        
        for ant in range(num_ants):
            if starting_vertex is None:
                current_vertex = random.randint(0, num_vertices - 1)  # Placing each ant in a vertex selected randomly
                starting_vertex_this_ant = current_vertex
            else:
                current_vertex = starting_vertex  # Place all ants at the source vertex
                starting_vertex_this_ant = starting_vertex
            visited = [False] * num_vertices
            visited[current_vertex] = True
            solution = [current_vertex]
            solution_cost = 0
            
            itr_ant = 0
            while True:
                probabilities = []
                denominator = 0

                for vertex in range(num_vertices):
                    if (((not visited[vertex]) and one_time_visit) or (not one_time_visit)) and (vertex != current_vertex):
                        pheromone_value = pheromone[current_vertex][vertex]
                        distance = graph[current_vertex][vertex]
                        probability = (pheromone_value ** alpha) * ((1.0 / distance) ** beta)
                        probabilities.append((vertex, probability))
                        denominator += probability
                    else:
                        probabilities.append((vertex, 0))

                probabilities = [(node, prob / denominator) for node, prob in probabilities]
                probabilities = [prob for (_, prob) in probabilities]
                next_vertex = random.choices(range(num_vertices), probabilities)[0]
                visited[next_vertex] = True
                solution.append(next_vertex)
                solution_cost += graph[current_vertex][next_vertex]
                current_vertex = next_vertex
                
                # Termination of ant
                if visit_all_vertices:
                    if len(np.unique(solution)) == num_vertices:
                        if TSP:
                            solution.append(starting_vertex_this_ant)
                            solution_cost += graph[current_vertex][starting_vertex_this_ant]
                        if stopping_vertex is None:
                            break
                        elif current_vertex == stopping_vertex:
                            break
                elif current_vertex == stopping_vertex:
                    break
                
                if (max_itr_ant is not None) and (itr_ant > max_itr_ant):
                    break
                
                itr_ant += 1
                
            # Update the best cost and the best solution (path)
            if solution_cost < best_cost:
                best_solution = solution
                best_cost = solution_cost

            solutions.append((solution, solution_cost))

        # Decrement (evaporate) the pheromones gradually
        for i in range(num_vertices):
            for j in range(num_vertices):
                pheromone[i][j] = (1.0 - pheromone_decay) * pheromone[i][j]

        # Update (strengthen) the pheromones gradually
        for solution, solution_cost in solutions:
            for i in range(len(solution) - 1):
                pheromone[solution[i]][solution[i + 1]] += 1.0 / solution_cost
                pheromone[solution[i + 1]][solution[i]] += 1.0 / solution_cost

        # Update the tqdm progress bar
        progress_bar.update(1)  # Increment the progress bar by 1 unit
        itr += 1
        
    progress_bar.close()
    
    return best_solution, best_cost

##############################################################################################################
############ Helper Functions ################################################################################
##############################################################################################################
def bound_solution_in_x_range(x: List[float], x_range: List[List[float]]) -> List[float]:
    for j in range(len(x)):
        if x[j] < x_range[j][0]:
            x[j] = x_range[j][0]
        elif x[j] > x_range[j][1]:
            x[j] = x_range[j][1]
    return x