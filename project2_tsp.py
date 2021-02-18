import networkx as nx
import osmnx as ox
from osmnx import distance as distance
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import csv
import random
import operator
import math
import pandas
import matplotlib.pyplot as plt

# ========================================
# Created by: Brian Bennett
# Last Modified by: Jake Gillenwater
# Course: CSCI-5260-940, Artificial Intelligence
# Date: 2/17/2021
# ========================================

# -- Set up the initial map area and save it as a networkx graph
ox.config(log_console=True, use_cache=True)
G = ox.graph_from_address(
    '1276 Gilbreath Drive, Johnson City, Washington County, Tennessee, 37614, United States',
    dist=8000, network_type='drive')

# Use this code to display a plot of the graph if desired. Note: You need to import matplotlib.pyplot as plt
# fig, ax = ox.plot_graph(G, edge_linewidth=3, node_size=0, show=False, close=False)
# plt.show()

# -- Genetic Algorithm Parameters
GENERATIONS = 1000
POPULATION_SIZE = 200
MUTATION_RATE = 0.1
DISPLAY_RATE = 100

# -- Set up Origin and Destination Points
origin_point = (36.3044549, -82.3632187)  # Start at ETSU
destination_point = (36.3044549, -82.3632187)  # End at ETSU
origin = ox.get_nearest_node(G, origin_point)
destination = ox.get_nearest_node(G, destination_point)
origin_node = (origin, G.nodes[origin])
destination_node = (destination, G.nodes[destination])

# -- Set up initial lists
points = []                 # The list of osmnx nodes that can be used for map plotting
generations = []            # A list of populations, ultimately of size GENERATIONS
population = []             # The current population of size POPULATION_SIZE
# Represented as a list of index values that correspond to the points list
chromosome = []


def plot_path(lat, long, origin_point, destination_point, fitness):
    """
    SOURCE: Modified from Priyam, Apurv (2020). https://towardsdatascience.com/find-and-plot-your-optimal-path-using-plotly-and-networkx-in-python-17e75387b873

    Given a list of latitudes and longitudes, origin
    and destination point, plots a path on a map

    Parameters
    ----------
    lat, long: list of latitudes and longitudes
    origin_point, destination_point: co-ordinates of origin
    and destination
    Returns
    -------
    Nothing. Only shows the map.
    """
    origin = (origin_point[1]["y"], origin_point[1]["x"])
    destination = (destination_point[1]["y"], destination_point[1]["x"])
    # adding the lines joining the nodes
    fig = go.Figure(go.Scattermapbox(
        name="Path",
        mode="lines",
        lon=long,
        lat=lat,
        marker=go.scattermapbox.Marker(
            size=9
        ),
        text=[str(i) for i in range(1, len(long))],
        line=dict(width=4.5, color='blue')))
    # adding source marker
    fig.add_trace(go.Scattermapbox(
        name="Source",
        mode="markers",
        lon=[origin[1]],
        lat=[origin[0]],
        marker={'size': 12, 'color': "red"}))

    # adding destination marker
    fig.add_trace(go.Scattermapbox(
        name="Destination",
        mode="markers",
        lon=[destination[1]],
        lat=[destination[0]],
        marker={'size': 12, 'color': 'green'},
        text=str(fitness)))

    # getting center for plots:
    lat_center = np.mean(lat)
    long_center = np.mean(long)
    # defining the layout using mapbox_style
    fig.update_layout(mapbox_style="stamen-terrain",
                      mapbox_center_lat=30, mapbox_center_lon=-80)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      mapbox={
                          'center': {'lat': lat_center,
                                     'lon': long_center},
                          'zoom': 12})
    fig.show()


def plot_ga():
    generation_values = []
    best = []
    worst = []
    gen = 1
    for g in generations:
        best_route = g[0]
        worst_route = g[POPULATION_SIZE - 1]
        best.append(best_route[1])
        worst.append(worst_route[1])
        generation_values.append(gen)
        gen = gen + 1
    data = {'Generations': generation_values, 'Best': best, 'Worst': worst}
    df = pandas.DataFrame(data)
    fig = px.line(df, x="Generations", y=[
                  "Best", "Worst"], title="Fitness Across Generations")
    fig.show()


def haversine(point1, point2):
    """
    Returns the Great Circle Distance between point 1 and point 2 in miles
    """
    return ox.distance.great_circle_vec(
        G.nodes[point1]['y'],
        G.nodes[point1]['x'],
        G.nodes[point2]['y'],
        G.nodes[point2]['x'],
        3963.1906)


def calculate_fitness(chromosome):
    """
    Fitness is the total route cost using the haversine distance.
    The GA should attempt to minimize the fitness; minimal fitness => best fitness
    """
    fitness = 0.0
    return [chromosome, fitness]


# initialize population
def initialize_population():
    """
    Initialize the population by creating POPULATION_SIZE chromosomes.
    Each chromosome represents the index of the point in the points list.
    Sorts the population by fitness and adds it to the generations list.
    """
    my_population = []

    generations.append(my_population)


def repopulate(gen):
    """
    Creates a new generation by repopulation based on the previous generation.
    Calls selection, crossover, and mutate to create a child chromosome. Calculates fitness
    and continues until the population is full. Sorts the population by fitness
    and adds it to the generations list.
    """
    my_population = []

    generations.append(my_population)


def selection(gen):
    """
    Choose two parents and return their chromosomes
    """
    parent1, parent2 = None, None
    return parent1, parent2


def crossover(p1, p2):
    """
    Strategy: ...
    """
    child = []

    return child


def mutate(chromosome):
    """
    Strategy: swap two pairs of points. Return the chromosome after mutation.
    """
    mutant_child = []
    return mutant_child


def run_ga():
    """
    Initialize and repopulate until you have reached the maximum generations
    """
    initialize_population()

    for gen in range(GENERATIONS - 1):  # Note, you already ran generation 1
        repopulate(gen + 1)
        if gen % DISPLAY_RATE == 0:
            # Print the generation, and the best (lowest) fitness score in the
            # population for that generation
            print("Generation Stuff")


def show_route(generation_number):
    """
    Gets the latitude and longitude points for the best route in generation_number
    """
    the_route = generations[generation_number][0][0]
    the_fitness = generations[generation_number][0][1]

    startend = [g for g in G.nodes(True) if g[0] == origin_node[0]][0]

    route = [startend[0]]
    lat = [startend[1]["y"]]
    long = [startend[1]["x"]]

    for p in the_route:
        node = [g for g in G.nodes(True) if g[0] == points[p][0]][0]
        route.append(node[0])
        lat.append(points[p][1]["y"])
        long.append(points[p][1]["x"])

    route.append(startend[0])
    lat.append(startend[1]["y"])
    long.append(startend[1]["x"])
    plot_path(lat, long, origin_node, destination_node, the_fitness)
    print("The fitness for generation", generation_number, "was", the_fitness)


def main():
    """
    Reads the csv file and then runs the genetic algorithm and displays results.
    """
    with open('addresses_geocoded.csv') as file:

        csvFile = csv.reader(file)

        for xy in csvFile:
            point_coordinates = (float(xy[0]), float(xy[1]))
            point = ox.get_nearest_node(G, point_coordinates)
            point_node = (point, G.nodes[point])
            if point_node not in points:
                points.append(point_node)

    print("There were", len(points), "unique points found.")

    run_ga()
    # show_route(0)
    # show_route(math.floor(GENERATIONS/2))
    # show_route(GENERATIONS-1)

    # plot_ga()


main()
