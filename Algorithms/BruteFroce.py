import numpy as np
import itertools
import time
import math

class BruteForceTSP:
    """
    Brute force solution for TSP
    """
    
    def __init__(self, distance_matrix):
        """
        Setup solver with distance info
        """
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.best_route = None
        self.min_distance = float('inf')
    
    def solve(self):
        """
        Check all routes to find shortest one
        """
        # Start and end at city 0
        cities = list(range(1, self.num_cities))
        self.min_distance = float('inf')
        self.best_route = None
        
        permutations = list(itertools.permutations(cities))
        
        for perm in permutations:
            current_route = [0] + list(perm) + [0]
            
            # Calculate route distance
            current_distance = self.calculate_route_distance(current_route)
            
            # Update best route
            if current_distance < self.min_distance:
                self.min_distance = current_distance
                self.best_route = current_route
        
        return self.best_route, self.min_distance
    
    def calculate_route_distance(self, route):
        """
        Total distance of route
        """
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.distance_matrix[route[i], route[i+1]]
        return total_distance
    
    def get_computational_complexity(self):
        """
        Algorithm complexity info
        """
        return {
            'time_complexity': f"O({self.num_cities}!)",
            'space_complexity': f"O({self.num_cities}!)",
            'max_feasible_cities': 11,
            'expected_permutations': math.factorial(self.num_cities - 1)
        }

def brute_force_tsp(distance_matrix):
    """
    Find shortest path
    """
    solver = BruteForceTSP(distance_matrix)
    return solver.solve()

def measure_execution_time(distance_matrix):
    """
    Measure execution time
    """
    start_time = time.time()
    solver = BruteForceTSP(distance_matrix)
    route, distance = solver.solve()
    end_time = time.time()
    
    execution_time = end_time - start_time
    return route, distance, execution_time
