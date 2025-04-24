import dimod
import neal
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

class QUBOToIsingTSPSolver:
    """
    Class that implements the QUBO→Ising workflow for solving TSP.
    """
    
    def __init__(self, distance_matrix):
        """
        Initialize the QUBO→Ising TSP solver.
        
        Args:
            distance_matrix: Distance matrix between cities
        """
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)
        self.best_route = None
        self.best_distance = float('inf')
    
    def create_qubo_tsp(self, penalty=None):
        """
        Create a QUBO model for the TSP problem.
        
        Args:
            penalty: Optional penalty factor
            
        Returns:
            Dictionary representing the QUBO matrix Q
        """
        N = self.n_cities
        
        # If not specified, set a sufficiently large penalty factor
        if penalty is None:
            max_dist = np.max(self.distance_matrix)
            penalty = 2 * N * max_dist
        
        Q = {}

        # Constraint 1: Each city is visited once (time slot constraints)
        for i in range(N):
            for t in range(N):
                Q[(i * N + t, i * N + t)] = -penalty
                for tp in range(t + 1, N):
                    Q[(i * N + t, i * N + tp)] = 2 * penalty

        # Constraint 2: Each time slot has exactly one city
        for t in range(N):
            for i in range(N):
                for ip in range(i + 1, N):
                    Q[(i * N + t, ip * N + t)] = 2 * penalty

        # Objective: Minimize total travel distance
        for i in range(N):
            for j in range(N):
                if i != j:
                    for t in range(N - 1):
                        Q[(i * N + t, j * N + t + 1)] = Q.get((i * N + t, j * N + t + 1), 0) + self.distance_matrix[i][j]
                    # Return to start city
                    Q[(i * N + N - 1, j * N + 0)] = Q.get((i * N + N - 1, j * N + 0), 0) + self.distance_matrix[i][j]

        return Q
    
    def create_ising_from_qubo(self, Q):
        """
        Convert a QUBO model to an Ising model.
        
        Args:
            Q: QUBO matrix
            
        Returns:
            BQM model in Ising format
        """
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
        return bqm.change_vartype(dimod.SPIN, inplace=False)
    
    def decode_solution(self, sample):
        """
        Decode a binary solution into a TSP route.
        
        Args:
            sample: Dictionary with the binary solution
            
        Returns:
            List with the decoded route
        """
        N = self.n_cities
        route = [-1] * N
        for key, value in sample.items():
            if value == 1:
                city = key // N
                time = key % N
                route[time] = city
        
        # Ensure all cities are assigned to a time slot
        if -1 in route or len(set(route)) != N:
            raise ValueError("Decoding error: not all time slots have been filled with a unique city")
        
        return route
    
    def force_valid_solution(self, sample):
        """
        Construct a valid route even if the solution is not optimal.
        
        Args:
            sample: Dictionary with the binary solution
            
        Returns:
            List with a valid route
        """
        N = self.n_cities
        route = [-1] * N
        cities_used = set()
        
        # First, assign cities based on the solution
        for key, value in sample.items():
            if value == 1:
                city = key // N
                time = key % N
                if route[time] == -1 and city not in cities_used:
                    route[time] = city
                    cities_used.add(city)
        
        # Fill empty slots with unused cities
        available_cities = set(range(N)) - cities_used
        for i in range(N):
            if route[i] == -1:
                if available_cities:
                    city = available_cities.pop()
                    route[i] = city
        
        # If there are still empty slots, fill them with cities (duplicates if necessary)
        remaining_slots = [i for i in range(N) if route[i] == -1]
        if remaining_slots:
            all_cities = list(range(N))
            for slot in remaining_slots:
                route[slot] = np.random.choice(all_cities)
        
        return route
    
    def calculate_route_cost(self, route):
        """
        Calculate the total cost of a route.
        
        Args:
            route: List with the route to evaluate
            
        Returns:
            The total cost of the route
        """
        cost = 0
        N = len(route)
        for i in range(N - 1):
            cost += self.distance_matrix[route[i]][route[i+1]]
        # Add cost to return to the start city
        cost += self.distance_matrix[route[-1]][route[0]]
        return cost
    
    def solve_with_qubo_to_ising(self, num_reads=1000, sweeps=1000):
        """
        Solve the TSP with the QUBO→Ising workflow.
        
        Args:
            num_reads: Number of reads for the solver
            sweeps: Number of sweeps for the simulated annealing
            
        Returns:
            Tuple (route, cost, execution_time)
        """
        start_time = time.time()
        
        # 1. Create the QUBO model
        print("Step 1: Creating QUBO model for TSP...")
        Q = self.create_qubo_tsp()
        
        # 2. Convert QUBO to Ising
        print("Step 2: Converting QUBO model to Ising...")
        ising_bqm = self.create_ising_from_qubo(Q)
        
        # 3. Solve the Ising model
        print("Step 3: Solving the Ising model...")
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(ising_bqm, num_reads=num_reads, num_sweeps=sweeps)
        
        # Convert the solution from spins to binary for decoding
        best_sample = {k: (v+1)//2 for k, v in sampleset.first.sample.items()}
        
        # 4. Decode the solution
        print("Step 4: Decoding the solution...")
        try:
            best_route = self.decode_solution(best_sample)
            self.best_route = best_route
            self.best_distance = self.calculate_route_cost(best_route)
            valid = True
        except ValueError as e:
            print(f"Error: {e}")
            # Try to construct the best possible route from the partial solution
            self.best_route = self.force_valid_solution(best_sample)
            self.best_distance = self.calculate_route_cost(self.best_route)
            valid = False
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\n--- Result of the QUBO→Ising process ---")
        print(f"Valid route: {valid}, Cost: {self.best_distance:.2f}, Time: {execution_time:.4f}s")
        
        return self.best_route, self.best_distance, execution_time
    
    def visualize_solution(self, coordinates, title="TSP Solution using QUBO→Ising"):
        """
        Visualize the TSP solution.
        
        Args:
            coordinates: City coordinates
            title: Chart title
            
        Returns:
            Matplotlib figure with the visualization
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot city points
        ax.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', s=100, zorder=2)
        
        # Ensure we have a route to plot
        if self.best_route is None:
            print("No route to visualize. Run solve_with_qubo_to_ising first.")
            return fig
        
        # Plot the route
        route = self.best_route
        for i in range(len(route)):
            j = (i + 1) % len(route)
            ax.plot([coordinates[route[i], 0], coordinates[route[j], 0]],
                    [coordinates[route[i], 1], coordinates[route[j], 1]],
                    'r-', alpha=0.7, zorder=1)
        
        # Add city labels
        for i, (x, y) in enumerate(coordinates):
            ax.annotate(f"{i}", (x, y), fontsize=12)
        
        ax.set_title(f"{title}\nTotal distance: {self.best_distance:.2f}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.grid(True)
        
        return fig

# Helper functions to maintain compatibility with existing code

def create_qubo_tsp(distances, penalty=None):
    """Create a QUBO model for the TSP problem."""
    solver = QUBOToIsingTSPSolver(distances)
    return solver.create_qubo_tsp(penalty)

def create_ising_tsp(distances, penalty=None):
    """Create an Ising model for TSP from QUBO."""
    solver = QUBOToIsingTSPSolver(distances)
    Q = solver.create_qubo_tsp(penalty)
    return solver.create_ising_from_qubo(Q)

def solve_tsp_qubo_to_ising(distances, num_reads=1000, sweeps=1000):
    """Solve the TSP with the QUBO→Ising workflow."""
    solver = QUBOToIsingTSPSolver(distances)
    return solver.solve_with_qubo_to_ising(num_reads, sweeps)

def qubo_to_ising_workflow(distances, coordinates=None, num_reads=1000, sweeps=1000):
    """
    Implement the QUBO→Ising workflow for TSP.
    
    Args:
        distances: Distance matrix
        coordinates: Coordinates (optional for visualization)
        num_reads: Number of reads for the solver
        sweeps: Number of sweeps for simulated annealing
        
    Returns:
        Dictionary with the process result
    """
    solver = QUBOToIsingTSPSolver(distances)
    route, cost, execution_time = solver.solve_with_qubo_to_ising(num_reads, sweeps)
    
    # Visualize if coordinates are available
    visualization = None
    if coordinates is not None:
        visualization = solver.visualize_solution(coordinates)
    
    return {
        'route': route,
        'cost': cost,
        'time': execution_time,
        'visualization': visualization
    }

def measure_execution_time(distances, coordinates=None, method='qubo_to_ising', num_reads=1000, sweeps=1000):
    """
    Main function to measure TSP execution time.
    
    Args:
        distances: Distance matrix
        coordinates: Coordinates (optional)
        method: 'qubo', 'ising' or 'qubo_to_ising'
        num_reads: Number of reads for the solver
        sweeps: Number of sweeps for simulated annealing
        
    Returns:
        route, distance, execution_time, visualization
    """
    solver = QUBOToIsingTSPSolver(distances)
    
    if method == 'qubo_to_ising':
        route, distance, execution_time = solver.solve_with_qubo_to_ising(num_reads, sweeps)
    else:
        raise ValueError("Only the 'qubo_to_ising' method is supported in this implementation")
    
    # Create visualization if coordinates were provided
    visualization = None
    if coordinates is not None:
        visualization = solver.visualize_solution(coordinates)
    
    return route, distance, execution_time, visualization

# Example usage
if __name__ == "__main__":
    print("Example of TSP solved with the QUBO→Ising workflow")
    
    # Create a small example
    distances = np.array([
        [0, 2, 9, 10],
        [1, 0, 6, 4],
        [15, 7, 0, 8],
        [6, 3, 12, 0]
    ])
    
    # Generate coordinates for visualization
    from sklearn.manifold import MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coordinates = mds.fit_transform(distances)
    
    # Create and use the solver
    solver = QUBOToIsingTSPSolver(distances)
    route, cost, time = solver.solve_with_qubo_to_ising()
    
    print(f"Route found: {route}")
    print(f"Total cost: {cost:.2f}")
    print(f"Execution time: {time:.4f} seconds")
    
    # Visualize the solution
    fig = solver.visualize_solution(coordinates)
    plt.show()
