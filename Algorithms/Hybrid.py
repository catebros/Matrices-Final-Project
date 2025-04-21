import numpy as np
import networkx as nx
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler, ExactSolver
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import matplotlib.pyplot as plt
import time
import random
from itertools import permutations, combinations
from scipy.spatial import distance_matrix as calc_distance_matrix

try:
    # Try to import D-Wave library
    from dwave.system import DWaveSampler, EmbeddingComposite
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    print("D-Wave libraries not available. Using simulated quantum annealing instead.")

class HybridTSPSolver:
    """Quantum-classical hybrid solver for the Traveling Salesman Problem."""
    
    def __init__(self, distance_matrix, use_quantum=True, max_quantum_size=20, token=None):
        """
        Initialize the hybrid TSP solver.
        
        Args:
            distance_matrix: Distance matrix between cities
            use_quantum: Whether to use quantum processing when available
            max_quantum_size: Maximum problem size for quantum computing
            token: D-Wave API token (optional)
        """
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)
        self.use_quantum = use_quantum and DWAVE_AVAILABLE
        self.max_quantum_size = max_quantum_size
        self.token = token
        self.quantum_accessible = False
        
        # Try to set up the quantum solver if requested
        if self.use_quantum and self.n_cities <= self.max_quantum_size and DWAVE_AVAILABLE:
            try:
                if self.token:
                    self.quantum_sampler = EmbeddingComposite(DWaveSampler(token=self.token))
                else:
                    self.quantum_sampler = EmbeddingComposite(DWaveSampler())
                self.quantum_accessible = True
                print("Successfully connected to D-Wave quantum computer")
            except Exception as e:
                print(f"Failed to connect to D-Wave: {e}")
                self.quantum_accessible = False
    
    def create_tsp_qubo(self, cities_subset=None):
        """
        Create a QUBO model for the TSP problem.
        
        Args:
            cities_subset: Optional list of city indices for subproblem
            
        Returns:
            BinaryQuadraticModel for the TSP problem
        """
        if cities_subset is None:
            n = self.n_cities
            distances = self.distance_matrix
        else:
            n = len(cities_subset)
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        distances[i, j] = self.distance_matrix[cities_subset[i], cities_subset[j]]
        
        # Decision variables: x[i,p] = 1 if city i is at position p
        # QUBO size: n^2 x n^2 (for complete problem)
        
        # Create a dictionary to store QUBO terms
        Q = {}
        
        # Normalize distances to improve numerical performance
        max_dist = np.max(distances)
        if max_dist > 0:
            distances = distances / max_dist
        
        # Penalty parameters
        A = 4.0  # Penalty for one-city-per-position constraints
        B = 4.0  # Penalty for one-position-per-city constraints
        
        # 1. Terms for route cost
        for i in range(n):
            for j in range(n):
                if i != j:
                    for p in range(n):
                        # Connection between consecutive cities in the route
                        p_next = (p + 1) % n
                        Q[((i, p), (j, p_next))] = distances[i, j]
        
        # 2. Constraint: Each position must have exactly one city
        for p in range(n):
            # Quadratic term: -A * sum_i(x[i,p]) + A * sum_i,j: i<j (x[i,p] * x[j,p])
            for i in range(n):
                Q[((i, p), (i, p))] = Q.get(((i, p), (i, p)), 0) - A
                for j in range(i+1, n):
                    Q[((i, p), (j, p))] = Q.get(((i, p), (j, p)), 0) + 2*A
        
        # 3. Constraint: Each city must be at exactly one position
        for i in range(n):
            # Quadratic term: -B * sum_p(x[i,p]) + B * sum_p,q: p<q (x[i,p] * x[i,q])
            for p in range(n):
                Q[((i, p), (i, p))] = Q.get(((i, p), (i, p)), 0) - B
                for q in range(p+1, n):
                    Q[((i, p), (i, q))] = Q.get(((i, p), (i, q)), 0) + 2*B
        
        # Convert to BQM format for the solver
        bqm = BinaryQuadraticModel.empty(dimod.BINARY)
        
        # Add terms to the BQM
        for (i, p), (j, q) in Q:
            if i == j and p == q:  # Linear term
                bqm.add_variable((i, p), Q[((i, p), (j, q))])
            else:  # Quadratic term
                bqm.add_interaction((i, p), (j, q), Q[((i, p), (j, q))])
        
        return bqm
    
    def solve_quantum(self, cities_subset=None):
        """
        Solve the TSP problem using a quantum approach.
        
        Args:
            cities_subset: Optional list of city indices for the subproblem
            
        Returns:
            Best route found, route distance
        """
        if cities_subset is None:
            n = self.n_cities
            subset_indices = list(range(n))
        else:
            n = len(cities_subset)
            subset_indices = cities_subset
        
        try:
            # Create the QUBO model
            bqm = self.create_tsp_qubo(cities_subset)
            
            # Decide which solver to use
            if self.quantum_accessible and n <= self.max_quantum_size:
                print(f"Using quantum solver for {n} cities")
                response = self.quantum_sampler.sample(bqm, num_reads=100)
            elif n <= 10:  # For small problems, use exact solution
                print(f"Using exact solver for {n} cities")
                response = ExactSolver().sample(bqm)
            else:
                print(f"Using simulated annealing for {n} cities")
                response = SimulatedAnnealingSampler().sample(
                    bqm, 
                    num_reads=1000,
                    beta_range=[0.1, 50],
                    num_sweeps=1000
                )
            
            # Get the best solution
            sample = response.first.sample
            
            # Convert the sample into a route
            route = self._sample_to_route(sample, n)
            
            # Check if the route is valid
            if not self._is_valid_route(route, n):
                print("Invalid route from quantum solver. Using nearest neighbor fallback.")
                route = self._nearest_neighbor_tsp(subset_indices)
        
        except Exception as e:
            print(f"Error in quantum solver: {e}")
            print("Using nearest neighbor fallback.")
            route = self._nearest_neighbor_tsp(subset_indices)
        
        # Convert the solution indices back to the original indices if necessary
        if cities_subset is not None:
            route = [cities_subset[i] for i in route]
        
        # Calculate the route distance
        distance = self._calculate_route_distance(route)
        return route, distance
    
    def _sample_to_route(self, sample, n):
        """Convert a QUBO sample to a TSP route."""
        route = [-1] * n
        for (city, pos), value in sample.items():
            if value == 1:
                route[pos] = city
        
        # Fix the route if there are unassigned positions
        if -1 in route:
            unused_cities = set(range(n)) - set(route)
            for i, city in enumerate(route):
                if city == -1:
                    route[i] = unused_cities.pop()
        
        return route
    
    def _is_valid_route(self, route, n):
        """Check if a route is valid (contains each city exactly once)."""
        if len(route) != n:
            return False
        
        city_count = {}
        for city in route:
            if city < 0 or city >= n:
                return False
            city_count[city] = city_count.get(city, 0) + 1
        
        return all(count == 1 for count in city_count.values()) and len(city_count) == n
    
    def _calculate_route_distance(self, route):
        """Calculate the total distance of a route."""
        distance = 0
        n = len(route)
        for i in range(n):
            j = (i + 1) % n
            distance += self.distance_matrix[route[i], route[j]]
        return distance
    
    def _nearest_neighbor_tsp(self, cities=None):
        """Nearest neighbor algorithm for TSP as fallback."""
        if cities is None:
            cities = list(range(self.n_cities))
        
        n = len(cities)
        unvisited = set(range(n))
        route = [0]  # Start from the first city
        unvisited.remove(0)
        
        # Build the tour
        while unvisited:
            current = route[-1]
            # Find the closest unvisited neighbor
            next_city = min(unvisited, key=lambda city: 
                            self.distance_matrix[cities[current], cities[city]])
            route.append(next_city)
            unvisited.remove(next_city)
        
        # Convert to original city indices
        return [cities[i] for i in route]
    
    def _two_opt_improvement(self, route):
        """Improve the route using the 2-opt algorithm."""
        improved = True
        best_distance = self._calculate_route_distance(route)
        
        while improved:
            improved = False
            for i in range(1, len(route) - 1):
                for j in range(i + 1, len(route)):
                    if j - i == 1:
                        continue  # Do not invert adjacent segments
                    
                    # Calculate the change in distance if we invert the segment
                    new_route = route.copy()
                    new_route[i:j+1] = reversed(new_route[i:j+1])
                    new_distance = self._calculate_route_distance(new_route)
                    
                    if new_distance < best_distance:
                        route = new_route
                        best_distance = new_distance
                        improved = True
                        break
                
                if improved:
                    break
        
        return route, best_distance
    
    def _divide_cities(self, k=2):
        """Divide cities into k clusters using k-means."""
        from sklearn.cluster import KMeans
        
        # Create a matrix of fictitious coordinates using MDS
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coordinates = mds.fit_transform(self.distance_matrix)
        
        # Apply k-means
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(coordinates)
        
        # Organize cities by cluster
        city_clusters = [[] for _ in range(k)]
        for i, cluster_id in enumerate(clusters):
            city_clusters[cluster_id].append(i)
        
        # Add connection points between clusters (closest cities between clusters)
        connection_points = []
        for i in range(k):
            for j in range(i+1, k):
                if city_clusters[i] and city_clusters[j]:
                    # Find the closest cities between clusters i and j
                    min_dist = float('inf')
                    best_pair = None
                    for ci in city_clusters[i]:
                        for cj in city_clusters[j]:
                            dist = self.distance_matrix[ci, cj]
                            if dist < min_dist:
                                min_dist = dist
                                best_pair = (ci, cj)
                    
                    if best_pair:
                        connection_points.append(best_pair)
        
        return city_clusters, connection_points, coordinates
    
    def solve_large_instance(self):
        """
        Solve large instances by dividing the problem into subproblems.
        """
        if self.n_cities <= self.max_quantum_size:
            # For small problems, solve directly
            route, distance = self.solve_quantum()
            return route, distance
        
        # Determine number of clusters based on problem size
        k = max(2, self.n_cities // self.max_quantum_size)
        
        # Divide into clusters
        clusters, connection_points, coordinates = self._divide_cities(k)
        
        # Solve each cluster
        cluster_routes = []
        for i, cluster in enumerate(clusters):
            if len(cluster) > 1:  # Only solve non-empty clusters with more than one city
                print(f"Solving cluster {i+1}/{k} with {len(cluster)} cities")
                route, _ = self.solve_quantum(cluster)
                cluster_routes.append(route)
            elif len(cluster) == 1:
                cluster_routes.append(cluster)  # Single city
        
        # Connect clusters into a single route
        complete_route = self._connect_clusters(cluster_routes, connection_points)
        
        # Apply 2-opt to improve the final solution
        improved_route, final_distance = self._two_opt_improvement(complete_route)
        
        return improved_route, final_distance
    
    def _connect_clusters(self, cluster_routes, connection_points):
        """
        Connect cluster routes into a single complete route.
        """
        if len(cluster_routes) == 1:
            return cluster_routes[0]
        
        # Create a graph to represent connections between clusters
        G = nx.Graph()
        
        # Add nodes (each cluster is a node)
        for i, route in enumerate(cluster_routes):
            G.add_node(i, route=route)
        
        # Add edges with weights based on distances between connection points
        for (city1, city2) in connection_points:
            # Find which clusters these cities belong to
            cluster1 = None
            cluster2 = None
            for i, route in enumerate(cluster_routes):
                if city1 in route:
                    cluster1 = i
                if city2 in route:
                    cluster2 = i
                if cluster1 is not None and cluster2 is not None:
                    break
            
            if cluster1 is not None and cluster2 is not None and cluster1 != cluster2:
                G.add_edge(cluster1, cluster2, weight=self.distance_matrix[city1, city2],
                           connection=(city1, city2))
        
        # Find a minimum spanning tree
        mst = nx.minimum_spanning_tree(G)
        
        # Perform a DFS traversal to get the order of clusters
        cluster_order = list(nx.dfs_preorder_nodes(mst, 0))
        
        # Connect the cluster routes according to the DFS order
        complete_route = []
        visited_clusters = set()
        
        for i in range(len(cluster_order)):
            current = cluster_order[i]
            next_cluster = cluster_order[(i + 1) % len(cluster_order)]
            
            # Get the route of the current cluster
            current_route = list(G.nodes[current]['route'])
            
            if current not in visited_clusters:
                # If there is a connection with the next cluster, rearrange the route
                if G.has_edge(current, next_cluster):
                    connection = G[current][next_cluster]['connection']
                    city1, city2 = connection
                    
                    # Rotate the route to end with the connection point
                    if city1 in current_route:
                        idx = current_route.index(city1)
                        current_route = current_route[idx:] + current_route[:idx]
                    
                complete_route.extend(current_route)
                visited_clusters.add(current)
        
        return complete_route
    
    def solve(self):
        """
        Main method to solve the TSP problem.
        
        Returns:
            route: The optimal route found
            distance: Total distance of the route
        """
        start_time = time.time()
        
        if self.n_cities <= self.max_quantum_size and self.quantum_accessible:
            print("Solving directly with quantum approach")
            route, distance = self.solve_quantum()
        else:
            print("Solving large instance with hybrid divide-and-conquer approach")
            route, distance = self.solve_large_instance()
        
        # Apply final 2-opt improvement
        route, distance = self._two_opt_improvement(route)
        
        execution_time = time.time() - start_time
        print(f"Total execution time: {execution_time:.2f} seconds")
        print(f"Total distance: {distance:.2f}")
        
        return route, distance, execution_time

def measure_execution_time(distance_matrix, coordinates=None, use_quantum=False):
    """
    Measure execution time of the hybrid TSP algorithm.
    
    Args:
        distance_matrix: Distance matrix between cities
        coordinates: City coordinates for visualization (optional)
        use_quantum: Whether to use quantum computing when available
        
    Returns:
        route: The best route found
        distance: The route distance
        execution_time: Execution time in seconds
        visualization: Process visualization
    """
    # Check that distance_matrix is not None
    if distance_matrix is None:
        raise ValueError("Distance matrix cannot be None")
    
    try:
        # Create and run the hybrid solver
        solver = HybridTSPSolver(distance_matrix, use_quantum=use_quantum)
        route, distance, execution_time = solver.solve()
        
        # Create visualization if coordinates were provided
        visualization = None
        if coordinates is not None:
            visualization = visualize_solution(coordinates, route)
        
    except Exception as e:
        print(f"Error in hybrid algorithm: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Create a fallback solution in case of error
        n = len(distance_matrix)
        route = list(range(n))
        
        # Calculate distance
        distance = 0
        for i in range(n):
            j = (i + 1) % n
            distance += distance_matrix[route[i], route[j]]
        
        execution_time = 0
        visualization = None
    
    return route, distance, execution_time, visualization

def visualize_solution(coordinates, route):
    """
    Create a visualization of the hybrid solution.
    
    Args:
        coordinates: City coordinates
        route: Solution route
        
    Returns:
        fig: Matplotlib figure with the visualization
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot city points
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', s=100, zorder=2)
    
    # Plot the route
    for i in range(len(route)):
        j = (i + 1) % len(route)
        ax.plot([coordinates[route[i], 0], coordinates[route[j], 0]],
                [coordinates[route[i], 1], coordinates[route[j], 1]],
                'r-', alpha=0.7, zorder=1)
    
    # Add city labels
    for i, (x, y) in enumerate(coordinates):
        ax.annotate(f"{i}", (x, y), fontsize=12)
    
    ax.set_title("Quantum-Classical Hybrid Solution")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True)
    
    return fig

# Additional required imports
import dimod
from sklearn.cluster import KMeans
from sklearn.manifold import MDS