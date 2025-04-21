import numpy as np
import time
import random

class GeneticTSP:
    def __init__(self, distance_matrix, population_size=100, generations=500,
                mutation_rate=0.02, crossover_rate=0.8, elitism_rate=0.1):
        """
        Initialize the genetic algorithm for TSP.
        
        Args:
            distance_matrix: Matrix of distances between cities
            population_size: Number of individuals in population
            generations: Maximum number of generations to evolve
            mutation_rate: Probability of mutation for each gene
            crossover_rate: Probability of crossover
            elitism_rate: Percentage of best individuals to preserve
        """
        self.distance_matrix = distance_matrix
        self.n_cities = distance_matrix.shape[0]
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        
        # Performance tracking
        self.best_history = []
        self.avg_history = []
    
    def initialize_population(self):
        """Initialize a random population of tours"""
        population = []
        for _ in range(self.population_size):
            # Generate random permutation (always start/end at city 0)
            individual = [0] + list(np.random.permutation(range(1, self.n_cities))) + [0]
            population.append(individual)
        return population
    
    def calculate_fitness(self, individual):
        """Calculate the fitness (inverse of total distance) of an individual"""
        total_distance = 0
        for i in range(len(individual) - 1):
            from_city = individual[i]
            to_city = individual[i + 1]
            total_distance += self.distance_matrix[from_city, to_city]
        
        # We want to maximize fitness, so take inverse of distance
        return 1.0 / total_distance if total_distance > 0 else float('inf')
    
    def calculate_distance(self, individual):
        """Calculate the total distance of a route"""
        total_distance = 0
        for i in range(len(individual) - 1):
            from_city = individual[i]
            to_city = individual[i + 1]
            total_distance += self.distance_matrix[from_city, to_city]
        return total_distance
    
    def tournament_selection(self, population, fitnesses, tournament_size=3):
        """Select an individual using tournament selection"""
        population_indices = list(range(len(population)))
        tournament_indices = random.sample(population_indices, tournament_size)
        
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
        
        return population[winner_idx]
    
    def ordered_crossover(self, parent1, parent2):
        """
        Perform ordered crossover (OX) between two parents.
        Preserves the relative order of cities from both parents.
        """
        # We exclude the first and last city (always city 0)
        size = len(parent1) - 2
        
        # Choose random start and end points for the crossover segment
        start, end = sorted(random.sample(range(size), 2))
        
        # Initialize child with -1s (placeholders)
        child = [-1] + [-1] * size + [-1]
        
        # Copy segment from parent1 to child
        for i in range(start + 1, end + 2):
            child[i] = parent1[i]
        
        # Fill remaining positions with cities from parent2 in order
        child_idx = 1
        for i in range(1, len(parent2) - 1):
            # Skip the crossover segment
            if child_idx == start + 1:
                child_idx = end + 2
            
            if child_idx >= len(child) - 1:
                break
            
            city = parent2[i]
            # Only add cities that aren't already in the child
            if city not in child:
                child[child_idx] = city
                child_idx += 1
        
        # Set first and last cities to 0
        child[0] = 0
        child[-1] = 0
        
        return child
    
    def mutate(self, individual):
        """
        Apply mutation to an individual with probability mutation_rate.
        Uses swap mutation.
        """
        # Skip first and last city (always 0)
        if random.random() < self.mutation_rate:
            # Choose two random positions to swap (excluding first and last)
            pos1, pos2 = random.sample(range(1, len(individual) - 1), 2)
            individual[pos1], individual[pos2] = individual[pos2], individual[pos1]
        
        return individual
    
    def two_opt_improvement(self, individual, max_iterations=20):
        """
        Apply 2-opt local search to improve a route.
        Tries to remove route crossings by reconnecting edges.
        """
        best_distance = self.calculate_distance(individual)
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(1, len(individual) - 2):
                for j in range(i + 1, len(individual) - 1):
                    # Skip adjacent edges
                    if j - i == 1:
                        continue
                    
                    # Calculate current edge distances
                    current_distance = (
                        self.distance_matrix[individual[i-1], individual[i]] +
                        self.distance_matrix[individual[j], individual[j+1]]
                    )
                    
                    # Calculate new edge distances if we reconnect
                    new_distance = (
                        self.distance_matrix[individual[i-1], individual[j]] +
                        self.distance_matrix[individual[i], individual[j+1]]
                    )
                    
                    # If reconnecting is better, perform 2-opt swap
                    if new_distance < current_distance:
                        # Reverse the segment between i and j
                        individual[i:j+1] = reversed(individual[i:j+1])
                        
                        new_total_distance = self.calculate_distance(individual)
                        if new_total_distance < best_distance:
                            best_distance = new_total_distance
                            improved = True
        
        return individual
    
    def evolve(self):
        """Run the genetic algorithm evolution process"""
        # Initialize population
        population = self.initialize_population()
        
        # Calculate initial fitness for each individual
        fitnesses = [self.calculate_fitness(ind) for ind in population]
        
        # Track the best individual
        best_individual = population[fitnesses.index(max(fitnesses))]
        best_fitness = max(fitnesses)
        
        # Main evolution loop
        for generation in range(self.generations):
            # Calculate number of elite individuals to preserve
            n_elite = max(1, int(self.elitism_rate * self.population_size))
            
            # Sort population by fitness (descending)
            sorted_indices = np.argsort(fitnesses)[::-1]
            
            # Create new population starting with elite individuals
            new_population = [population[idx] for idx in sorted_indices[:n_elite]]
            
            # Create rest of new population through selection, crossover, mutation
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self.ordered_crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                child = self.mutate(child)
                
                # Occasionally apply local search to improve individuals
                if random.random() < 0.1:  # 10% chance for local search
                    child = self.two_opt_improvement(child)
                
                new_population.append(child)
            
            # Update population
            population = new_population
            
            # Recalculate fitness
            fitnesses = [self.calculate_fitness(ind) for ind in population]
            
            # Update best individual if improved
            generation_best_fitness = max(fitnesses)
            generation_best_idx = fitnesses.index(generation_best_fitness)
            
            if generation_best_fitness > best_fitness:
                best_fitness = generation_best_fitness
                best_individual = population[generation_best_idx]
            
            # Store statistics
            self.best_history.append(1.0 / best_fitness)
            self.avg_history.append(1.0 / np.mean(fitnesses))
        
        # Return the best individual found and its distance
        best_distance = 1.0 / best_fitness
        return best_individual, best_distance, self.best_history, self.avg_history

def solve_tsp(distance_matrix, population_size=100, generations=500):
    """
    Solve TSP using the genetic algorithm.
    
    Args:
        distance_matrix: Matrix of distances between cities
        population_size: Size of population
        generations: Number of generations to evolve
        
    Returns:
        best_route: Best route found
        best_distance: Distance of best route
        best_history: History of best distances
        avg_history: History of average distances
    """
    # Adapt parameters based on problem size
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
    
    # Initialize and run algorithm
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
    Measure execution time of genetic algorithm.
    
    Args:
        distance_matrix: Matrix of distances between cities
        population_size: Size of population
        generations: Number of generations to evolve
        
    Returns:
        best_route: Best route found
        best_distance: Distance of best route
        execution_time: Time taken in seconds
        best_history: History of best distances
        avg_history: History of average distances
    """
    start_time = time.time()
    
    best_route, best_distance, best_history, avg_history = solve_tsp(
        distance_matrix, population_size, generations
    )
    
    execution_time = time.time() - start_time
    
    return best_route, best_distance, execution_time, best_history, avg_history
