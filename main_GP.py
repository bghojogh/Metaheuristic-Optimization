# DEAP: DISTRIBUTED EVOLUTIONARY ALGORITHMS IN PYTHON
# https://deap.readthedocs.io/en/master/
# https://deap.readthedocs.io/en/master/tutorials/advanced/gp.html
# pip install deap
import operator
import random, yaml
import numpy as np
from deap import base, creator, tools, gp
from typing import Optional, List, Tuple

def create_random_dataset(n_samples: Optional[int] = 100, n_dimensions: Optional[int] = 5) -> Tuple[np.ndarray, List[float]]:
    # Generate random dataset
    np.random.seed(42)
    X = np.random.rand(n_samples, n_dimensions)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] + np.random.normal(0, 0.1, n_samples)
    return X, y

def deap_genetic_programming(functions: List[str], X: np.ndarray, y: List[float], max_expression_depth: Optional[int] = None):
    # Create DEAP types for fitness and individuals
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # Define the primitive set with only mathematical operations
    n_dimensions = X.shape[1]
    pset = gp.PrimitiveSet("MAIN", arity=n_dimensions)  # arity equals to the number of dimensions

    # Define mathematical operations (arity: the number of arguments or operands that a function or operator takes)
    if 'addition' in functions:
        pset.addPrimitive(operator.add, arity=2)
    if 'subtraction' in functions:
        pset.addPrimitive(operator.sub, arity=2)
    if 'multiplication' in functions:
        pset.addPrimitive(operator.mul, arity=2)
    if 'division' in functions:
        pset.addPrimitive(operator.truediv, arity=2)  # Use truediv for regular division

    # Define the evaluation function (MSE as the fitness function)
    def evaluate(individual, max_depth=max_expression_depth) -> float:
        func = gp.compile(expr=individual, pset=pset)

        # Replace variables in the expression using the mapping
        y_pred = [func(*X[i]) for i in range(len(X))]
        mse = np.mean((y_pred - y) ** 2)

        # Penalize expressions that exceed the specified max_depth
        if max_depth is not None:
            if len(individual) > max_depth:
                mse += (len(individual) - max_depth)  # You can adjust this penalty factor

        return mse

    # Create a toolbox for GP operations
    toolbox = base.Toolbox()
    if max_expression_depth is not None:
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_expression_depth)
    else:
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    return toolbox

def format_expression(individual):
    stack = list(individual)
    output = []
    while stack:
        item = stack.pop()
        if isinstance(item, gp.Primitive):
            func_str = item.name
            args = []
            for _ in range(item.arity):
                args.append(output.pop())
            # formatted_args = f"({args[0]} {func_str} {args[1]})"
            if func_str == "add":
                formatted_args = f"({args[0]} + {args[1]})"
            elif func_str == "sub":
                formatted_args = f"({args[0]} - {args[1]})"
            elif func_str == "mul":
                formatted_args = f"({args[0]} * {args[1]})"
            elif func_str == "truediv":
                formatted_args = f"({args[0]} / {args[1]})"
            output.append(formatted_args)
        elif isinstance(item, gp.Terminal):
            output.append(item.name)
    return output[0]

def main_genetic_programming(n_samples: int, n_dimensions: int, functions: List[str], population: Optional[int] = 300, n_iterations: Optional[int] = 10,
                        crossover_prob: Optional[float] = 0.7, mutation_prob: Optional[float] = 0.3, max_expression_depth: Optional[int] = None):
    # Set the random seed
    random.seed(42)

    # Create dataset
    X, y = create_random_dataset(n_samples=n_samples, n_dimensions=n_dimensions)

    # Make the modules in DEAP library
    toolbox = deap_genetic_programming(functions=functions, X=X, y=y, max_expression_depth=max_expression_depth)
    
    # Create an initial population of individuals
    population = toolbox.population(n=population)
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    fitnesses = [(i,) for i in fitnesses]
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Perform the evolutionary loop
    CXPB, MUTPB, NGEN = crossover_prob, mutation_prob, n_iterations
    for gen in range(NGEN):
        print(f"Generation {gen + 1}/{NGEN}")
        
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))  # Clones to make sure that the genetic operations are applied to the copies, not originals.
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)  # Creates new individuals
                del child1.fitness.values  # Re-evaluation needed!
                del child2.fitness.values  # Re-evaluation needed!
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the offspring individuals
        fitnesses = list(map(toolbox.evaluate, offspring))
        fitnesses = [(i,) for i in fitnesses]
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        
        # Replace the old population by the offspring
        population[:] = offspring
        
        # Gather all the fitnesses in the population
        fits = [ind.fitness.values[0] for ind in population]
        
        # Print the best individual in this generation
        best_ind = tools.selBest(population, 1)[0]
        print(f"Best individual's fitness (cost): {best_ind.fitness.values[0]}")
    
    best_individual = tools.selBest(population, 1)[0]
    print("\nBest individual found:")
    
    # Format the best individual as a human-readable expression
    expression = format_expression(best_individual)
    print(expression)

if __name__ == "__main__":
    with open('./config/config_GP.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main_genetic_programming(n_samples=config['n_samples'], n_dimensions=config['n_dimensions'], functions=config['functions'],
                            population=config['population'], n_iterations=config['n_iterations'],
                            crossover_prob=config['crossover_prob'], mutation_prob=config['mutation_prob'], max_expression_depth=config['max_expression_depth'])
