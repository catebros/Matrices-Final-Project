import numpy as np
import time

class NearestNeighborTSP:
    """
    Implements the Nearest Neighbor heuristic for TSP.
    """
    
    def __init__(self, distance_matrix):
        """
        Initialize the solver with a distance matrix.
        """
        self.distance_matrix = distance_matrix
        self.n_cities = distance_matrix.shape[0]
        self.best_route = None
        self.best_distance = float('inf')
    
    def solve_from_city(self, start_city=0):
        """
        Solve TSP starting from a specific city.
        """
        route = [start_city]
        unvisited = set(range(self.n_cities))
        unvisited.remove(start_city)
        current_city = start_city
        
        while unvisited:
            next_city = min(unvisited, key=lambda city: self.distance_matrix[current_city][city])
            route.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        
        route.append(start_city)
        total_distance = self.calculate_route_distance(route)
        
        return route, total_distance
    
    def calculate_route_distance(self, route):
        """
        Calculate the distance of a route.
        """
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.distance_matrix[route[i]][route[i+1]]
        return total_distance
    
    def solve_multi_start(self):
        """
        Run NN starting from each city and return the best result.
        """
        self.best_distance = float('inf')
        self.best_route = None
        
        for start_city in range(self.n_cities):
            route, distance = self.solve_from_city(start_city)
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_route = route
        
        return self.best_route, self.best_distance
    
    def apply_two_opt_improvement(self, route, max_iterations=100):
        """
        Improve a route using 2-opt.
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
                        continue
                    
                    new_route = improved_route.copy()
                    new_route[i:j+1] = reversed(new_route[i:j+1])
                    new_distance = self.calculate_route_distance(new_route)
                    
                    if new_distance < best_distance:
                        improved_route = new_route
                        best_distance = new_distance
                        improved = True
                        break
                
                if improved:
                    break
        
        return improved_route, best_distance
    
    def solve_with_improvement(self):
        """
        Solve TSP using NN and improve with 2-opt.
        """
        initial_route, initial_distance = self.solve_multi_start()
        improved_route, improved_distance = self.apply_two_opt_improvement(initial_route)
        
        return improved_route, improved_distance

def nearest_neighbor_tsp(distance_matrix, start_city=0):
    """
    Solve TSP using NN starting from a specific city.
    """
    solver = NearestNeighborTSP(distance_matrix)
    return solver.solve_from_city(start_city)

def multi_start_nearest_neighbor(distance_matrix):
    """
    Run NN starting from each city and return the best result.
    """
    solver = NearestNeighborTSP(distance_matrix)
    return solver.solve_multi_start()

def measure_execution_time(distance_matrix, multi_start=True, apply_improvement=False):
    """
    Measure execution time of NN algorithm.
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
    Shows how bad NN can be compared to optimal.
    """
    n = distance_matrix.shape[0]
    
    return f"Theoretical worst-case: up to {np.log(n):.2f} times optimal"