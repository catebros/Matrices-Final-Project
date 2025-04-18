import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import BruteFroce as bf
import Hybrid as h
import Quantum as q
import Heuristic as ht
import time

def generate_data(n_cities):
    """Generates random coordinates and a distance matrix for n cities"""
    city_coordinates = np.random.rand(n_cities, 2) * 100

    distance_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                distance_matrix[i, j] = np.sqrt(
                    (city_coordinates[i, 0] - city_coordinates[j, 0])**2 +
                    (city_coordinates[i, 1] - city_coordinates[j, 1])**2
                )

    return city_coordinates, distance_matrix

def show_route(coordinates, route=None):
    """Displays the cities and the route in a plot"""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', s=100, zorder=2)

    for i, (x, y) in enumerate(coordinates):
        ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')

    if route is not None:
        for i in range(len(route) - 1):
            ax.plot([coordinates[route[i], 0], coordinates[route[i+1], 0]],
                    [coordinates[route[i], 1], coordinates[route[i+1], 1]],
                    'r-', zorder=1)
    else:
        print("No route provided.")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.set_title("Traveling Salesman Problem")

    return fig

def main():
    st.title("Traveling Salesman Problem (TSP)")

    st.sidebar.header("Configuration")

    n_cities = st.sidebar.slider("Number of cities", 4, 15, 5)

    show_matrix = st.sidebar.checkbox("Show distance matrix", False)

    if st.sidebar.button("Generate new data"):
        st.session_state.coordinates, st.session_state.distance_matrix = generate_data(n_cities)
        st.session_state.matrix_generated = True
        st.session_state.results_generated = False

    if 'matrix_generated' not in st.session_state:
        st.session_state.coordinates, st.session_state.distance_matrix = generate_data(n_cities)
        st.session_state.matrix_generated = True
        st.session_state.results_generated = False

    if 'last_n_cities' not in st.session_state or st.session_state.last_n_cities != n_cities:
        st.session_state.coordinates, st.session_state.distance_matrix = generate_data(n_cities)
        st.session_state.matrix_generated = True
        st.session_state.results_generated = False
        st.session_state.last_n_cities = n_cities

    st.write(f"Current configuration: {n_cities} cities")

    st.subheader("City Map")
    fig = show_route(st.session_state.coordinates)
    st.pyplot(fig)

    if show_matrix:
        st.subheader("Distance Matrix")
        st.write(st.session_state.distance_matrix)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Run Brute Force"):
            with st.spinner("Running brute force algorithm..."):
                route_bf, dist_bf = run_brute_force(st.session_state.distance_matrix)
                st.session_state.route_bf = route_bf
                st.session_state.dist_bf = dist_bf
                st.session_state.results_brute_force = True

    with col2:
        if st.button("Run Greedy Algorithm"):
            with st.spinner("Running greedy algorithm..."):
                route_greedy, dist_greedy = run_greedy_algorithm(st.session_state.distance_matrix)
                st.session_state.route_greedy = route_greedy
                st.session_state.dist_greedy = dist_greedy
                st.session_state.results_greedy = True

    with col3:
        if st.button("Run Genetic Algorithm"):
            with st.spinner("Running genetic algorithm..."):
                route_gen, dist_gen = run_genetic(st.session_state.distance_matrix)
                st.session_state.route_gen = route_gen
                st.session_state.dist_gen = dist_gen
                st.session_state.results_genetic = True

    st.subheader("Results")

    result_tabs = st.tabs(["Brute Force", "Greedy Algorithm", "Genetic Algorithm"])

    with result_tabs[0]:
        if 'results_brute_force' in st.session_state and st.session_state.results_brute_force:
            st.write(f"Total distance: {st.session_state.dist_bf:.2f}")
            st.write(f"Route: {st.session_state.route_bf}")
            fig_bf = show_route(st.session_state.coordinates, st.session_state.route_bf)
            st.pyplot(fig_bf)
        else:
            st.info("Run the brute force algorithm to see the results")

    with result_tabs[1]:
        if 'results_greedy' in st.session_state and st.session_state.results_greedy:
            st.write(f"Total distance: {st.session_state.dist_greedy:.2f}")
            st.write(f"Route: {st.session_state.route_greedy}")
            fig_greedy = show_route(st.session_state.coordinates, st.session_state.route_greedy)
            st.pyplot(fig_greedy)
        else:
            st.info("Run the greedy algorithm to see the results")

    with result_tabs[2]:
        if 'results_genetic' in st.session_state and st.session_state.results_genetic:
            st.write(f"Total distance: {st.session_state.dist_gen:.2f}")
            st.write(f"Route: {st.session_state.route_gen}")
            fig_gen = show_route(st.session_state.coordinates, st.session_state.route_gen)
            st.pyplot(fig_gen)
        else:
            st.info("Run the genetic algorithm to see the results")

if __name__ == "__main__":
    main()