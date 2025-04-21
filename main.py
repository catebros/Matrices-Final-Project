import numpy as np
import matplotlib.pyplot as plt
import time
import streamlit as st
from matplotlib.animation import FuncAnimation
import pandas as pd
import seaborn as sns

# Import algorithm modules (updated to reflect the new location)
from Algorithms.GeneticHeuristic import measure_execution_time as genetic_tsp
from Algorithms.NNHeuristic import measure_execution_time as nn_tsp
from Algorithms.BruteFroce import measure_execution_time as bf_tsp
from Algorithms.Hybrid import measure_execution_time as hybrid_tsp

# Set page configuration
st.set_page_config(page_title="TSP Solver", layout="wide", initial_sidebar_state="expanded")

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
    
    # Initialize session state for storing results
    if "results" not in st.session_state:
        st.session_state.results = {}
    
    # Sidebar for algorithm selection and parameters
    with st.sidebar:
        st.header("Algorithm Settings")
        
        # Problem settings
        st.subheader("Problem Settings")
        n_cities = st.slider("Number of Cities", min_value=5, max_value=50, value=10)
        random_seed = st.number_input("Random Seed", value=42, help="Seed for reproducibility")
        
        # Create/load distance matrix
        if st.button("Generate Random Problem"):
            st.session_state.distance_matrix, st.session_state.points = create_random_distance_matrix(
                n_cities, seed=random_seed
            )
            st.success(f"Generated random problem with {n_cities} cities")
            
            # Clear previous results when generating a new problem
            st.session_state.results = {}
        
        # Algorithm selection
        st.subheader("Select Algorithm")
        algorithm = st.selectbox(
            "Algorithm",
            ["Genetic Algorithm", "Nearest Neighbor", "Brute Force", "Hybrid"]
        )
        
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
            if n_cities > 10:
                st.warning(f"Warning: Brute Force approach is infeasible for problems with more than 10 cities. Current size: {n_cities} cities.")
            
            params = {}
            
        elif algorithm == "Hybrid":
            st.subheader("Hybrid Parameters")
            use_quantum = st.checkbox("Use Quantum Computing for Small Clusters", value=True)
            
            params = {
                "use_quantum": use_quantum
            }
        
        # Run algorithm button
        run_button = st.button("Run Algorithm")
    
    # Main area - display results
    if hasattr(st.session_state, 'distance_matrix') and hasattr(st.session_state, 'points'):
        # Display current problem
        st.subheader("Current Problem")
        st.write(f"Number of Cities: {len(st.session_state.points)}")
        
        # Visualization of city positions
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(st.session_state.points[:, 0], st.session_state.points[:, 1], s=100)
        for i, (x, y) in enumerate(st.session_state.points):
            ax.annotate(f"{i}", (x, y), fontsize=12)
        ax.set_title("City Positions")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.grid(True)
        st.pyplot(fig)
        
        # Run the selected algorithm if button is clicked
        if run_button:
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
                    
                elif algorithm == "Hybrid":
                    status_text.text("Running Hybrid Algorithm...")
                    try:
                        route, distance, execution_time, visualization = hybrid_tsp(
                            distance_matrix, coordinates=points, **params
                        )
                        
                        # Check that route is not None before storing the results
                        if route is None:
                            route = list(range(len(distance_matrix)))  # Create a default route
                            distance = float('inf')
                            st.warning("Hybrid algorithm returned an invalid route. Using default route instead.")
                        
                        # Store results
                        st.session_state.results[algorithm] = {
                            "route": route,
                            "distance": distance,
                            "time": execution_time,
                            "visualization": visualization if visualization is not None else None,
                            "n_cities": len(points),
                            "params": params
                        }
                    except TypeError as e:
                        # Provide a more descriptive error message
                        st.error(f"Error in hybrid algorithm: {str(e)}. Try with a smaller number of cities.")
                        return  # Exit the function to handle the error gracefully
                
                progress_bar.progress(100)
                status_text.text(f"Algorithm completed in {execution_time:.4f} seconds!")
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
                    
                    # Special visualization for hybrid results
                    if algo_name == "Hybrid" and "visualization" in result:
                        st.subheader("Cluster Visualization")
                        hybrid_fig = result["visualization"]
                        st.pyplot(hybrid_fig)
            
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