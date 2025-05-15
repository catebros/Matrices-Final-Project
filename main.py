import numpy as np
import matplotlib.pyplot as plt
import time
import streamlit as st
from matplotlib.animation import FuncAnimation
import pandas as pd
import seaborn as sns
from streamlit.components.v1 import html
import streamlit.components.v1 as components
import math


# Import algorithm modules (updated to reflect the new location)
from Algorithms.GeneticHeuristic import measure_execution_time as genetic_tsp
from Algorithms.NNHeuristic import measure_execution_time as nn_tsp
from Algorithms.BruteFroce import measure_execution_time as bf_tsp
from Algorithms.tsp_solver import qubo_to_ising_workflow

# Display imports
from Data.map import GreatBritainMap
from Data.simulation import Simulation

# Set page configuration
st.set_page_config(page_title="TSP Solver", layout="wide", initial_sidebar_state="expanded")

def create_distance_matrix_from_coords(coords):
    """
    coords: array-like of shape (n,2) with (latitude, longitude).
    Returns symmetric matrix of great-circle distances in kilometers.
    """
    n = len(coords)
    D = np.zeros((n, n))
    R = 6371.0  # Earth radius in km
    for i in range(n):
        lat1, lon1 = coords[i]
        phi1, lam1 = math.radians(lat1), math.radians(lon1)
        for j in range(i+1, n):
            lat2, lon2 = coords[j]
            phi2, lam2 = math.radians(lat2), math.radians(lon2)
            dphi = phi2 - phi1
            dlam = lam2 - lam1
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
            d = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            D[i, j] = D[j, i] = d
    return D

def create_random_distance_matrix(n_cities, seed=None):
    """Create a random symmetric distance matrix for testing."""
    if seed is not None:
        np.random.seed(seed)
    
    # Create random points in 2D space
    points = np.random.rand(n_cities, 2)
    
    # Calculate Euclidean distances between all pairs of points
    distance_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(i+1, n_cities):
            distance = np.sqrt(((points[i] - points[j]) ** 2).sum())
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetric matrix
    
    return distance_matrix, points

def plot_route(points, route, distance, title="Best Route"):
    """Plot the route on a 2D map."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], c='blue', s=100)
    
    # Plot route
    route_points = points[route]
    ax.plot(route_points[:, 0], route_points[:, 1], 'r-', alpha=0.7)
    
    # Add city labels
    for i, (x, y) in enumerate(points):
        ax.annotate(f"{i}", (x, y), fontsize=12)
    
    ax.set_title(f"{title}\nTotal Distance: {distance:.2f}")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True)
    
    return fig

def plot_convergence(best_history, avg_history=None, title="Convergence History"):
    """Plot the convergence history of the algorithm."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = list(range(len(best_history)))
    ax.plot(generations, best_history, 'b-', label="Best Distance")
    
    if avg_history is not None:
        ax.plot(generations, avg_history, 'r-', alpha=0.7, label="Average Distance")
    
    ax.set_title(title)
    ax.set_xlabel("Generation / Iteration")
    ax.set_ylabel("Distance")
    ax.grid(True)
    ax.legend()
    
    return fig

def compare_algorithms(results):
    """Compare algorithm performance based on results."""
    if not results:
        st.warning("No algorithms have been run yet.")
        return
    
    # Create comparison dataframes
    algorithms = list(results.keys())
    distances = [results[alg].get("distance", float('inf')) for alg in algorithms]
    times = [results[alg].get("time", 0) for alg in algorithms]
    
    # Create comparison figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot distance comparison
    axs[0].bar(algorithms, distances, color='skyblue')
    axs[0].set_title("Distance Comparison")
    axs[0].set_ylabel("Total Distance")
    axs[0].grid(True, axis='y')
    for i, v in enumerate(distances):
        axs[0].text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    # Plot time comparison
    axs[1].bar(algorithms, times, color='salmon')
    axs[1].set_title("Execution Time Comparison")
    axs[1].set_ylabel("Time (seconds)")
    axs[1].grid(True, axis='y')
    for i, v in enumerate(times):
        axs[1].text(i, v + 0.1, f"{v:.2f}s", ha='center')
    
    plt.tight_layout()
    return fig

def create_performance_table(results):
    """Create a dataframe with performance metrics."""
    if not results:
        return None
    
    data = {
        "Algorithm": [],
        "Distance": [],
        "Time (s)": [],
        "Cities": [],
        "Parameters": []
    }
    
    for algo, result in results.items():
        data["Algorithm"].append(algo)
        data["Distance"].append(f"{result.get('distance', 'N/A'):.2f}")
        data["Time (s)"].append(f"{result.get('time', 'N/A'):.4f}")
        data["Cities"].append(result.get("n_cities", "N/A"))
        
        # Format algorithm-specific parameters
        params = result.get("params", {})
        param_str = ", ".join(f"{k}: {v}" for k, v in params.items())
        data["Parameters"].append(param_str)
    
    return pd.DataFrame(data)

def main():
    """Main function to run the Streamlit app."""
    st.title("Traveling Salesman Problem Solver")
    
    gb_map = GreatBritainMap()
    sim = Simulation(city_df=gb_map.get_dataset())

    # map_html = gb_map.display_map()

    # html(map_html, height=600, width=450)
    
    # Initialize session state for storing results
    if "results" not in st.session_state:
        st.session_state.results = {}
    
    # Sidebar for algorithm selection and parameters
    with st.sidebar:
        st.header("Algorithm Settings")
        
        st.subheader("UK Cities Problem")
        full_uk_df = gb_map.get_dataset()
        max_uk_cities = len(full_uk_df)
        n_cities_uk = st.slider("Number of UK Cities to test on", min_value=5, max_value=max_uk_cities, value=10)
        
        generate_cities_button = st.button("Generate UK Problem")
          
        
        # Algorithm selection
        st.subheader("Select Algorithm")
        algorithm = st.selectbox(
            "Algorithm",
            ["Genetic Algorithm", "Nearest Neighbor", "Brute Force", "QUBO-Ising"]
        )
        
        # Sample Increase
        st.subheader("Algorithm Sample Increase")  
        st.text("Running the simulation will test the algorithm with sample sizes from 1 to 50, showing how its behavior changes as the sample size increases linearly.")
        simulation_button = st.button("Run Simulation")
        if simulation_button:
            pass
        
        # Algorithm-specific parameters
        if algorithm == "Genetic Algorithm":
            st.subheader("Genetic Algorithm Parameters")
            population_size = st.slider("Population Size", 10, 500, 100)
            generations = st.slider("Generations", 10, 1000, 200)
            
            params = {
                "population_size": population_size,
                "generations": generations
            }
            
        elif algorithm == "Nearest Neighbor":
            st.subheader("Nearest Neighbor Parameters")
            multi_start = st.checkbox("Use Multi-Start Approach", value=True)
            apply_improvement = st.checkbox("Apply 2-opt Improvement", value=False)
            
            params = {
                "multi_start": multi_start,
                "apply_improvement": apply_improvement
            }
            
        elif algorithm == "Brute Force":
            st.subheader("Brute Force Parameters")
            st.write("Brute Force uses exact algorithm with no parameters to tune.")
            
            # Warn about problem size
            if n_cities_uk > 10:
                st.warning(f"Warning: Brute Force approach is infeasible for problems with more than 10 cities. Current size: {n_cities_uk} cities.")
            
            params = {}
            
        elif algorithm == "QUBO-Ising":
            st.subheader("Quantum Parameters")
            num_reads = st.slider("Number of Reads", 10, 5000, 1000)
            sweeps = st.slider("Number of Sweeps", 10, 5000, 1000)
            
            # Warning for large problems
            if n_cities_uk > 20:
                st.warning(f"Quantum methods may be inefficient for problems with more than 20 cities. Current size: {n_cities_uk} cities.")
            
            params = {
                "num_reads": num_reads,
                "sweeps": sweeps
            }
        
        # Run algorithm button
        run_button = st.button("Run Algorithm")
    
    
    
    if generate_cities_button:
            gb_map.generate_uk_map(n_cities_uk)
            st.session_state.map_html = gb_map.display_map()
            html(st.session_state.map_html, height=600, width=450)
            
            sampled_df = gb_map.get_dataset()[["Latitude", "Longitude"]].to_numpy()
            distance_matrix = create_distance_matrix_from_coords(sampled_df)
            
            st.session_state.distance_matrix = distance_matrix
            st.session_state.points = sampled_df
            st.success(f"Loaded {n_cities_uk} cities from UK dataset")
            st.session_state.results = {}

    
    
    # Main area - display results
    if hasattr(st.session_state, 'distance_matrix') and hasattr(st.session_state, 'points'):
        # Display current problem
        st.subheader("Current Problem")
        st.write(f"Number of Cities: {len(st.session_state.points)}")
        
        # Visualization of city positions
        # fig, ax = plt.subplots(figsize=(8, 8))
        # ax.scatter(st.session_state.points[:, 0], st.session_state.points[:, 1], s=100)
        # for i, (x, y) in enumerate(st.session_state.points):
        #     ax.annotate(f"{i}", (x, y), fontsize=12)
        # ax.set_title("City Positions")
        # ax.set_xlabel("X Coordinate")
        # ax.set_ylabel("Y Coordinate")
        # ax.grid(True)
        # st.pyplot(fig)
                
        # if generate_cities_button:
        #     gb_map.generate_uk_map(n_cities_uk)
        #     st.session_state.map_html = gb_map.display_map()
        #     html(st.session_state.map_html, height=600, width=450)
            
        #     sampled_df = gb_map.get_dataset()[["Latitude", "Longitude"]].to_numpy()
        #     distance_matrix = create_distance_matrix_from_coords(sampled_df)
            
        #     st.session_state.distance_matrix = distance_matrix
        #     st.session_state.points = sampled_df
        #     st.success(f"Loaded {n_cities_uk} cities from UK dataset")
        #     st.session_state.results = {}

        
        if simulation_button:
            sim.run_all(
                max_cities=max_uk_cities,
                step=5,
                repeats=1
            )
            perf_fig = sim.plot_performance()
            st.pyplot(perf_fig)
        
        # Run the selected algorithm if button is clicked
        if run_button:
            html(st.session_state.map_html, height=600, width=450)
                
            st.subheader(f"Running {algorithm}...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            distance_matrix = st.session_state.distance_matrix
            points = st.session_state.points
            
            try:
                if algorithm == "Genetic Algorithm":
                    status_text.text("Running Genetic Algorithm...")
                    route, distance, execution_time, best_history, avg_history = genetic_tsp(
                        distance_matrix, **params
                    )
                    
                    # Store results
                    st.session_state.results[algorithm] = {
                        "route": route,
                        "distance": distance,
                        "time": execution_time,
                        "best_history": best_history,
                        "avg_history": avg_history,
                        "n_cities": len(points),
                        "params": params
                    }
                
                elif algorithm == "Nearest Neighbor":
                    status_text.text("Running Nearest Neighbor Algorithm...")
                    route, distance, execution_time = nn_tsp(
                        distance_matrix, **params
                    )
                    
                    # Store results
                    st.session_state.results[algorithm] = {
                        "route": route,
                        "distance": distance,
                        "time": execution_time,
                        "n_cities": len(points),
                        "params": params
                    }
                    
                elif algorithm == "Brute Force":
                    status_text.text("Running Brute Force Algorithm...")
                    route, distance, execution_time = bf_tsp(
                        distance_matrix
                    )
                    
                    # Store results
                    st.session_state.results[algorithm] = {
                        "route": route,
                        "distance": distance,
                        "time": execution_time,
                        "n_cities": len(points),
                        "params": params
                    }
                
                elif algorithm == "QUBO-Ising":
                    status_text.text("Running QUBO Ising workflow...")
                    result = qubo_to_ising_workflow(
                        distance_matrix, coordinates=points, **params
                    )
                    
                    # Store results
                    st.session_state.results[algorithm] = {
                        "route": result['route'],
                        "distance": result['cost'],
                        "time": result['time'],
                        "visualization": result['visualization'],
                        "n_cities": len(points),
                        "params": params
                    }
                
                # Update progress and show completion message
                progress_bar.progress(100)
                
                # Update with the correct execution time based on the algorithm
                if algorithm == "QUBO-Ising":
                    execution_time = result['time']
                
                status_text.text(f"Algorithm completed in {execution_time:.4f} seconds!")
                
                # Show success message with the appropriate distance
                if algorithm == "QUBO-Ising":
                    st.success(f"Found route with distance: {result['cost']:.2f}")
                else:
                    st.success(f"Found route with distance: {distance:.2f}")
                
            except Exception as e:
                st.error(f"Error running algorithm: {str(e)}")
                # Add debugging information
                import traceback
                st.error(f"Error details: {traceback.format_exc()}")
        
        # Display results if available 
        if st.session_state.results:
            st.header("Algorithm Results")
            
            # Create tabs for each algorithm result
            tabs = st.tabs(list(st.session_state.results.keys()) + ["Compare"])
            
            for i, (algo_name, result) in enumerate(st.session_state.results.items()):
                with tabs[i]:
                    st.subheader(f"{algo_name} Results")
                    
                    # Display route visualization
                    if "route" in result and "distance" in result:
                        if "visualization" in result and result["visualization"] is not None:
                            st.pyplot(result["visualization"])
                        else:
                            route_fig = plot_route(
                                st.session_state.points, result["route"], result["distance"],
                                title=f"Best Route found by {algo_name}"
                            )
                            st.pyplot(route_fig)
                        
                        # Display route as list
                        st.write(f"Route: {result['route']}")
                        st.write(f"Distance: {result['distance']:.2f}")
                        st.write(f"Execution time: {result['time']:.4f} seconds")
                    
                    # Display convergence history if available
                    if "best_history" in result:
                        hist_fig = plot_convergence(
                            result["best_history"],
                            result.get("avg_history"),
                            title=f"Convergence History - {algo_name}"
                        )
                        st.pyplot(hist_fig)
            
            # Comparison tab
            with tabs[-1]:
                st.subheader("Algorithm Comparison")
                
                # Performance table
                st.write("### Performance Metrics")
                perf_table = create_performance_table(st.session_state.results)
                if perf_table is not None:
                    st.table(perf_table)
                
                # Comparison charts
                comp_fig = compare_algorithms(st.session_state.results)
                st.pyplot(comp_fig)
                
                st.write("""                
                ### Analysis Notes
                - **Distance**: Lower is better - indicates more optimal route
                - **Time**: Lower is better - indicates faster computation
                - Different algorithms have different tradeoffs between solution quality and computation time
                - Problem characteristics (size, structure) can significantly impact algorithm performance
                """)
    else:
        st.info("Please generate a random problem using the sidebar")

if __name__ == "__main__":
    main()