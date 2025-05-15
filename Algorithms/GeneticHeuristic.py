import numpy as np
import time
import random

class GeneticTSP:
    def __init__(self, distance_matrix, population_size=100, generations=500,
                mutation_rate=0.02, crossover_rate=0.8, elitism_rate=0.1):
        """
        Args:
            distance_matrix: Distances
            population_size: Routes count
            generations: Iterations
            mutation_rate: Mutation chance
            crossover_rate: Crossover chance
            elitism_rate: Keep best
        """
        self.distance_matrix = distance_matrix
        self.n_cities = distance_matrix.shape[0]
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        
        # History
        self.best_history = []
        self.avg_history = []
    
    def initialize_population(self):
        """Random starting routes"""
        population = []
        for _ in range(self.population_size):
            # Start/end at 0
            individual = [0] + list(np.random.permutation(range(1, self.n_cities))) + [0]
            population.append(individual)
        return population
    
    def calculate_fitness(self, individual):
        """Get route quality"""
        total_distance = 0
        for i in range(len(individual) - 1):
            from_city = individual[i]
            to_city = individual[i + 1]
            total_distance += self.distance_matrix[from_city, to_city]
        
        # Inverse distance as fitness
        return 1.0 / total_distance if total_distance > 0 else float('inf')
    
    def calculate_distance(self, individual):
        """Get total route distance"""
        total_distance = 0
        for i in range(len(individual) - 1):
            from_city = individual[i]
            to_city = individual[i + 1]
            total_distance += self.distance_matrix[from_city, to_city]
        return total_distance
    
    def tournament_selection(self, population, fitnesses, tournament_size=3):
        """Select route via tournament"""
        population_indices = list(range(len(population)))
        tournament_indices = random.sample(population_indices, tournament_size)
        
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
        
        return population[winner_idx]
    
    def ordered_crossover(self, parent1, parent2):
        """Combine parent routes into child route"""
        size = len(parent1) - 2
        
        # Random start and end points
        start, end = sorted(random.sample(range(size), 2))
        
        child = [-1] + [-1] * size + [-1]
        
        # Copy segment from first parent
        for i in range(start + 1, end + 2):
            child[i] = parent1[i]
        
        child_idx = 1
        for i in range(1, len(parent2) - 1):
            if child_idx == start + 1:
                child_idx = end + 2
            
            if child_idx >= len(child) - 1:
                break
            
            city = parent2[i]
            if city not in child:
                child[child_idx] = city
                child_idx += 1
        
        # Ensure route starts and ends at city 0
        child[0] = 0
        child[-1] = 0
        
        return child
    
    def mutate(self, individual):
        """Randomly swap cities in route"""
        # Skip first and last city
        if random.random() < self.mutation_rate:
            pos1, pos2 = random.sample(range(1, len(individual) - 1), 2)
            individual[pos1], individual[pos2] = individual[pos2], individual[pos1]
        
        return individual
    
    def two_opt_improvement(self, individual, max_iterations=20):
        """Improve route by removing crossings"""
        best_distance = self.calculate_distance(individual)
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(1, len(individual) - 2):
                for j in range(i + 1, len(individual) - 1):
                    if j - i == 1:
                        continue
                    
                    current_distance = (
                        self.distance_matrix[individual[i-1], individual[i]] +
                        self.distance_matrix[individual[j], individual[j+1]]
                    )
                    
                    # New edge distances if swapped
                    new_distance = (
                        self.distance_matrix[individual[i-1], individual[j]] +
                        self.distance_matrix[individual[i], individual[j+1]]
                    )
                    
                    # Swap if improvement
                    if new_distance < current_distance:
                        individual[i:j+1] = reversed(individual[i:j+1])
                        
                        new_total_distance = self.calculate_distance(individual)
                        if new_total_distance < best_distance:
                            best_distance = new_total_distance
                            improved = True
        
        return individual
    
    def evolve(self):
        """Run genetic algorithm"""
        # Start with random population
        population = self.initialize_population()
        
        fitnesses = [self.calculate_fitness(ind) for ind in population]
        
        # Track best route
        best_individual = population[fitnesses.index(max(fitnesses))]
        best_fitness = max(fitnesses)
        
        # Run evolution for specified generations
        for generation in range(self.generations):
            n_elite = max(1, int(self.elitism_rate * self.population_size))
            
            sorted_indices = np.argsort(fitnesses)[::-1]
            
            new_population = [population[idx] for idx in sorted_indices[:n_elite]]
            
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                # Combine parents into child route
                if random.random() < self.crossover_rate:
                    child = self.ordered_crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                child = self.mutate(child)
                
                if random.random() < 0.1:  # 10% chance for local search
                    child = self.two_opt_improvement(child)
                
                new_population.append(child)
            
            # Replace old population
            population = new_population
            
            fitnesses = [self.calculate_fitness(ind) for ind in population]
            
            # Update best route if better found
            generation_best_fitness = max(fitnesses)
            generation_best_idx = fitnesses.index(generation_best_fitness)
            
            if generation_best_fitness > best_fitness:
                best_fitness = generation_best_fitness
                best_individual = population[generation_best_idx]
            
            # Record progress
            self.best_history.append(1.0 / best_fitness)
            self.avg_history.append(1.0 / np.mean(fitnesses))
        
        # Return best route and distance
        best_distance = 1.0 / best_fitness
        return best_individual, best_distance, self.best_history, self.avg_history

def solve_tsp(distance_matrix, population_size=100, generations=500):
    """
    Solve TSP using genetic algorithm
    
    Args:
        distance_matrix: City distances
        population_size: Number of routes
        generations: Number of iterations
        
    Returns:
        best_route: Shortest route found
        best_distance: Distance of best route
        best_history: Best distance improvements
        avg_history: Average distance improvements
    """
    # Adjust parameters based on number of cities
    n_cities = distance_matrix.shape[0]
    
    if n_cities <= 10:
        mutation_rate = 0.02
        crossover_rate = 0.8
    elif n_cities <= 20:
        mutation_rate = 0.03
        crossover_rate = 0.7
    else:
        mutation_rate = 0.04
        crossover_rate = 0.6
    
    # Setup and run algorithm
    ga = GeneticTSP(
        distance_matrix=distance_matrix,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        elitism_rate=0.1
    )
    
    return ga.evolve()

def measure_execution_time(distance_matrix, population_size=100, generations=500):
    """
    Measure algorithm runtime
    
    Args:
        distance_matrix: City distances
        population_size: Number of routes
        generations: Number of iterations
        
    Returns:
        best_route: Shortest route found
        best_distance: Distance of best route
        execution_time: Runtime in seconds
        best_history: Best distance improvements
        avg_history: Average distance improvements
    """
    start_time = time.time()
    
    best_route, best_distance, best_history, avg_history = solve_tsp(
        distance_matrix, population_size, generations
    )
    
    execution_time = time.time() - start_time
    
    return best_route, best_distance, execution_time, best_history, avg_history
