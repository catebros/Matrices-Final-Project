import numpy as np
import itertools
import time
import math

class BruteForceTSP:
    """
    Class that implements the brute force approach to solve the TSP.
    """
    
    def __init__(self, distance_matrix):
        """
        Initialize the Brute Force TSP solver.
        
        Args:
            distance_matrix: Matrix of distances between cities
        """
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.best_route = None
        self.min_distance = float('inf')
    
    def solve(self):
        """
        Solves the TSP using brute force approach by checking all possible permutations.
        
        Returns:
            best_route: The optimal route as a list of city indices
            min_distance: The minimum total distance
        """
        # Consider all cities except the first one (index 0)
        # as we start and end at city 0
        cities = list(range(1, self.num_cities))
        
        # Initialize with infinite values
        self.min_distance = float('inf')
        self.best_route = None
        
        # Generate all possible permutations of cities
        permutations = list(itertools.permutations(cities))
        
        for perm in permutations:
            # Add the starting and ending city (city 0)
            current_route = [0] + list(perm) + [0]
            
            # Calculate the total distance for this route
            current_distance = self.calculate_route_distance(current_route)
            
            # Update the best route if we find a better one
            if current_distance < self.min_distance:
                self.min_distance = current_distance
                self.best_route = current_route
        
        return self.best_route, self.min_distance
    
    def calculate_route_distance(self, route):
        """
        Calculate the total distance of a given route.
        
        Args:
            route: List of cities in the order they are visited
            
        Returns:
            total_distance: The total distance of the route
        """
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.distance_matrix[route[i], route[i+1]]
        return total_distance
    
    def get_computational_complexity(self):
        """
        Returns information about the computational complexity of the algorithm.
        
        Returns:
            complexity_info: Dictionary with complexity information
        """
        return {
            'time_complexity': f"O({self.num_cities}!)",
            'space_complexity': f"O({self.num_cities}!)",
            'max_feasible_cities': 11,  # Practical limit for most computers
            'expected_permutations': math.factorial(self.num_cities - 1)
        }

# Legacy functions for backward compatibility

def brute_force_tsp(distance_matrix):
    """
    Solves the Traveling Salesman Problem (TSP) using brute force approach.
    
    Args:
        distance_matrix: Distance matrix between cities.
        
    Returns:
        best_route: The optimal route as a list of city indices.
        min_distance: The minimum total distance.
    """
    solver = BruteForceTSP(distance_matrix)
    return solver.solve()

def measure_execution_time(distance_matrix):
    """
    Measures the execution time of the brute force TSP algorithm.
    
    Args:
        distance_matrix: Distance matrix between cities.
        
    Returns:
        route: The optimal route.
        distance: The minimum distance.
        execution_time: Time taken to execute in seconds.
    """
    start_time = time.time()
    solver = BruteForceTSP(distance_matrix)
    route, distance = solver.solve()
    end_time = time.time()
    
    execution_time = end_time - start_time
    return route, distance, execution_time
