import numpy as np
import time

class NearestNeighborTSP:
    """
    Class that implements the Nearest Neighbor heuristic for TSP.
    """
    
    def __init__(self, distance_matrix):
        """
        Initialize the Nearest Neighbor solver.
        
        Args:
            distance_matrix: Matrix of distances between cities
        """
        self.distance_matrix = distance_matrix
        self.n_cities = distance_matrix.shape[0]
        self.best_route = None
        self.best_distance = float('inf')
    
    def solve_from_city(self, start_city=0):
        """
        Solves the TSP using the Nearest Neighbor heuristic from a specific starting city.
        
        Args:
            start_city: The city to start the tour from (default: 0)
            
        Returns:
            route: The found route as a list of city indices
            total_distance: The total distance of the route
        """
        # Initialize variables
        route = [start_city]
        unvisited = set(range(self.n_cities))
        unvisited.remove(start_city)
        
        # Start from the given city and find the nearest neighbor
        current_city = start_city
        
        # Visit all cities
        while unvisited:
            next_city = min(unvisited, key=lambda city: self.distance_matrix[current_city][city])
            route.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        
        # Return to the starting city to complete the tour
        route.append(start_city)
        
        # Calculate the total distance
        total_distance = self.calculate_route_distance(route)
        
        return route, total_distance
    
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
            total_distance += self.distance_matrix[route[i]][route[i+1]]
        return total_distance
    
    def solve_multi_start(self):
        """
        Runs the Nearest Neighbor algorithm starting from each city and returns the best result.
        
        Returns:
            best_route: The best route found
            best_distance: The distance of the best route
        """
        self.best_distance = float('inf')
        self.best_route = None
        
        # Try starting from each city
        for start_city in range(self.n_cities):
            route, distance = self.solve_from_city(start_city)
            
            # Update best solution if better
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_route = route
        
        return self.best_route, self.best_distance
    
    def apply_two_opt_improvement(self, route, max_iterations=100):
        """
        Apply 2-opt local search to improve a route.
        
        Args:
            route: Initial route to improve
            max_iterations: Maximum number of improvement iterations
            
        Returns:
            improved_route: The improved route
            improved_distance: The distance of the improved route
        """
        improved_route = route.copy()
        best_distance = self.calculate_route_distance(improved_route)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(1, len(improved_route) - 2):
                for j in range(i + 1, len(improved_route) - 1):
                    if j - i == 1:
                        continue  # Skip adjacent edges
                    
                    # Try 2-opt swap: reverse the segment between i and j
                    new_route = improved_route.copy()
                    new_route[i:j+1] = reversed(new_route[i:j+1])
                    new_distance = self.calculate_route_distance(new_route)
                    
                    if new_distance < best_distance:
                        improved_route = new_route
                        best_distance = new_distance
                        improved = True
                        break  # Restart with the improved route
                
                if improved:
                    break
        
        return improved_route, best_distance
    
    def solve_with_improvement(self):
        """
        Solves TSP using Nearest Neighbor and then improves with 2-opt.
        
        Returns:
            best_route: The best improved route
            best_distance: The distance of the best route
        """
        # First get the initial solution using multi-start
        initial_route, initial_distance = self.solve_multi_start()
        
        # Then improve using 2-opt
        improved_route, improved_distance = self.apply_two_opt_improvement(initial_route)
        
        return improved_route, improved_distance

# Legacy functions that use the class internally - for backward compatibility

def nearest_neighbor_tsp(distance_matrix, start_city=0):
    """
    Solves the TSP using the Nearest Neighbor heuristic.
    
    Args:
        distance_matrix: Distance matrix between cities
        start_city: The city to start the tour from (default: 0)
        
    Returns:
        route: The found route as a list of city indices
        total_distance: The total distance of the route
    """
    solver = NearestNeighborTSP(distance_matrix)
    return solver.solve_from_city(start_city)

def multi_start_nearest_neighbor(distance_matrix):
    """
    Runs the Nearest Neighbor algorithm starting from each city and returns the best result.
    
    Args:
        distance_matrix: Distance matrix between cities
        
    Returns:
        best_route: The best route found
        best_distance: The distance of the best route
    """
    solver = NearestNeighborTSP(distance_matrix)
    return solver.solve_multi_start()

def measure_execution_time(distance_matrix, multi_start=True, apply_improvement=False):
    """
    Measures the execution time of the nearest neighbor TSP algorithm.
    
    Args:
        distance_matrix: Distance matrix between cities
        multi_start: Whether to try starting from each city (default: True)
        apply_improvement: Whether to apply 2-opt improvement (default: False)
        
    Returns:
        route: The best route found
        distance: The total distance of the route
        execution_time: Time taken to execute in seconds
    """
    start_time = time.time()
    
    solver = NearestNeighborTSP(distance_matrix)
    
    if apply_improvement:
        route, distance = solver.solve_with_improvement()
    elif multi_start:
        route, distance = solver.solve_multi_start()
    else:
        route, distance = solver.solve_from_city()
        
    execution_time = time.time() - start_time
    
    return route, distance, execution_time

def find_worst_case_improvement(distance_matrix):
    """
    Analyzes the gap between nearest neighbor and optimal solution (if available).
    For demonstration purposes only.
    
    Args:
        distance_matrix: Distance matrix between cities
        
    Returns:
        improvement_percentage: Estimated percentage that the solution could be improved
    """
    # This is a placeholder - in a real implementation, you might
    # compare against known optimal solutions or use bounds
    n = distance_matrix.shape[0]
    
    # Theoretical worst-case for NN is O(log n) times optimal
    # Just for demonstration - this is not an actual measure
    return f"Theoretical worst-case: up to {np.log(n):.2f} times optimal"