# Traveling Salesman Problem Solver

This project implements and compares different algorithms to solve the Traveling Salesman Problem (TSP).

## Implemented Algorithms

1. **Genetic Algorithm** - An evolutionary technique that generates solutions through selection, crossover, and mutation operators.
3. **Nearest Neighbor** - A greedy algorithm that always moves to the closest unvisited city.
4. **Brute Force** - Tests all possible permutations to find the optimal solution.
5. **Hybrid** - Combines classical and quantum techniques to solve larger-scale problems.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit interface to interact with the algorithms:

```bash
streamlit run main.py
```

### Interface Features

- Generate random TSP problems with customizable number of cities
- Select different algorithms to solve the problem
- Adjust algorithm-specific parameters
- Visualize results and compare algorithm performance
- View execution times and solution quality metrics

## Algorithm Details

### Genetic Algorithm
- Uses populations of candidate solutions that evolve over generations
- Implements selection based on route distance (fitness)
- Applies crossover and mutation to explore the solution space
- Parameters: population size, number of generations, mutation rate

### Nearest Neighbor
- Fast heuristic algorithm with O(n²) time complexity
- Can use multi-start approach (starting from each city)
- Optional 2-opt improvement for better solutions
- Good balance between speed and solution quality

### Brute Force
- Guarantees optimal solution by checking all permutations
- Limited to small problems (≤11 cities) due to factorial time complexity
- Serves as a baseline for evaluating other algorithms

### Hybrid Quantum-Classical
- Divides large problems into smaller subproblems
- Uses quantum or simulated quantum computing for subproblems
- Combines solutions using minimum spanning tree approach
- Applies 2-opt local improvement to refine final solutions

## Performance Considerations

- **Solution Quality**: Brute Force > Hybrid > Genetic > Nearest Neighbor
- **Execution Speed**: Nearest Neighbor > Genetic > Hybrid > Brute Force
- **Scalability**: Hybrid and Genetic algorithms scale better to larger problems
- **Parameter Sensitivity**: Genetic algorithm requires the most tuning

## Project Structure

- `main.py` - Main Streamlit application interface
- `Algorithms/` - Implementation of all algorithms
  - `GeneticHeuristic.py` - Genetic algorithm implementation
  - `NNHeuristic.py` - Nearest Neighbor implementation
  - `BruteFroce.py` - Brute Force implementation
  - `Hybrid.py` - Hybrid Quantum-Classical implementation
