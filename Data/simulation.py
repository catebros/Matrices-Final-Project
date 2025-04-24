import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Import the TSP algorithms' execution functions
from Algorithms.GeneticHeuristic import measure_execution_time as genetic_tsp
from Algorithms.NNHeuristic import measure_execution_time as nn_tsp
from Algorithms.BruteFroce import measure_execution_time as bf_tsp
from Algorithms.Hybrid import measure_execution_time as hybrid_tsp


class TestResult:
    """
    Stores the performance of a single algorithm run.
    Attributes:
        algorithm_name (str): Name of the algorithm.
        number_of_cities (int): Number of cities in the problem.
        time_taken (float): Execution time in seconds.
    """
    def __init__(self, algorithm_name: str, number_of_cities: int, time_taken: float):
        self.algorithm_name = algorithm_name
        self.number_of_cities = number_of_cities
        self.time_taken = time_taken


class Simulation:
    """
    Runs performance tests for multiple TSP algorithms over increasing problem sizes.
    """
    def __init__(self, city_df, random_state: int = 42):
        # Load the UK cities dataset
        self.city_df = city_df
        self.results: list[TestResult] = []
        self.random_state = random_state

    @staticmethod
    def generate_distance_matrix(coords: np.ndarray) -> np.ndarray:
        """
        Constructs a symmetric Euclidean distance matrix for given coordinates.
        """
        n = coords.shape[0]
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(coords[i] - coords[j])
                mat[i, j] = dist
                mat[j, i] = dist
        return mat

    def save_test_result(self, algorithm_name: str, number_of_cities: int, time_taken: float) -> TestResult:
        """
        Creates and stores a TestResult.
        """
        result = TestResult(algorithm_name, number_of_cities, time_taken)
        self.results.append(result)
        return result

    def run_all(
        self,
        max_cities: int = 49,
        step: int = 1,
        repeats: int = 1
    ) -> list[TestResult]:
        """
        Runs each algorithm over problem sizes [2, 2+step, ..., max_cities].
        Skips Brute Force for more than 10 cities.
        """
        algorithms = {
            'Genetic Algorithm': genetic_tsp,
            'Nearest Neighbor': nn_tsp,
            'Brute Force': bf_tsp,
            'Hybrid': hybrid_tsp
        }

        for n in range(2, max_cities + 1, step):
            sampled = self.city_df.sample(n=n, random_state=self.random_state)
            coords = sampled[['Latitude', 'Longitude']].to_numpy()
            dist_mat = self.generate_distance_matrix(coords)

            for algo_name, algo_func in algorithms.items():
                # Skip brute force for large problem sizes
                if algo_name == 'Brute Force' and n > 10:
                    print(f"Skipping Brute Force for {n} cities (too large)")
                    continue
                for r in range(repeats):
                    print(f"Running {algo_name} on {n} cities (run {r + 1}/{repeats})...")
                    try:
                        out = algo_func(dist_mat)
                        if isinstance(out, tuple) and len(out) >= 3:
                            exec_time = float(out[2])
                        else:
                            exec_time = float(out)
                    except Exception as e:
                        print(f"Error in {algo_name} with {n} cities: {e}")
                        exec_time = float('inf')
                    self.save_test_result(algo_name, n, exec_time)

        return self.results

    def plot_performance(self) -> None:
        """
        Plots execution time vs. number of cities for each algorithm.
        """
        # Organize results into a DataFrame
        import pandas as pd

        df = pd.DataFrame([
            {
                'Algorithm': res.algorithm_name,
                'Cities': res.number_of_cities,
                'Time': res.time_taken
            }
            for res in self.results
        ])

        # Pivot for plotting
        pivot = df.pivot(index='Cities', columns='Algorithm', values='Time')

        # Plot
        plt.figure(figsize=(10, 6))
        for algo in pivot.columns:
            plt.plot(pivot.index, pivot[algo], marker='o', label=algo)

        plt.xlabel('Number of Cities')
        plt.ylabel('Execution Time (s)')
        plt.title('TSP Algorithm Performance vs. Problem Size')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()