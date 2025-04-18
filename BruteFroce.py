import numpy as np
import itertools
import time

def brute_force_tsp(distance_matrix):
    """
    Resuelve el problema del Viajante de Comercio (TSP) mediante fuerza bruta.
    
    Args:
        distance_matrix: Matriz de distancias entre ciudades.
        
    Returns:
        best_route: La ruta óptima como lista de índices de ciudades.
        min_distance: La distancia total mínima.
    """
    num_cities = distance_matrix.shape[0]
    
    # Consideramos todas las ciudades excepto la primera (índice 0)
    # ya que empezamos y terminamos en la ciudad 0
    cities = list(range(1, num_cities))
    
    # Inicializamos con valores infinitos
    min_distance = float('inf')
    best_route = None
    
    # Generamos todas las permutaciones posibles de ciudades
    permutations = list(itertools.permutations(cities))
    
    for perm in permutations:
        # Añadimos la ciudad inicial y final (ciudad 0)
        current_route = [0] + list(perm) + [0]
        
        # Calculamos la distancia total de esta ruta
        current_distance = 0
        for i in range(len(current_route) - 1):
            current_distance += distance_matrix[current_route[i], current_route[i+1]]
        
        # Actualizamos la mejor ruta si encontramos una mejor
        if current_distance < min_distance:
            min_distance = current_distance
            best_route = current_route
    
    return best_route, min_distance

def run_tsp_example():
    """
    Ejecuta un ejemplo del algoritmo TSP con una matriz de distancia de ejemplo.
    """
    # Ejemplo de matriz de distancias (5 ciudades)
    distance_matrix = np.array([
        [0, 10, 15, 20, 25],
        [10, 0, 35, 25, 30],
        [15, 35, 0, 30, 10],
        [20, 25, 30, 0, 40],
        [25, 30, 10, 40, 0]
    ])
    
    print("Matriz de distancias:")
    print(distance_matrix)
    
    # Medimos el tiempo de ejecución
    start_time = time.time()
    
    # Resolvemos el TSP
    best_route, min_distance = brute_force_tsp(distance_matrix)
    
    end_time = time.time()
    
    print("\nRuta óptima:", best_route)
    print("Distancia mínima:", min_distance)
    print("Tiempo de ejecución: {:.6f} segundos".format(end_time - start_time))
    print("\nAdvertencia: Este algoritmo tiene complejidad O(n!), por lo que solo es práctico para un número pequeño de ciudades (≤ 11).")

if __name__ == "__main__":
    run_tsp_example()
