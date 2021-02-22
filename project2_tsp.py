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
import time

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
POPULATION_SIZE = 400
MUTATION_RATE = 0.8
DISPLAY_RATE = 100
RANDOM_SEED = time.time_ns()
RANDOM = random.Random(RANDOM_SEED)

# -- Set up Origin and Destination Points
origin_point = (36.3044549, -82.3632187)  # Start at ETSU
destination_point = (36.3044549, -82.3632187)  # End at ETSU
origin = ox.get_nearest_node(G, origin_point)
origin_id = origin
destination = ox.get_nearest_node(G, destination_point)
destination_id = destination
origin_node = (origin, G.nodes[origin])
destination_node = (destination, G.nodes[destination])

# -- Set up initial lists
points = []                 # The list of osmnx nodes that can be used for map plotting
# * Points format = [(nodeId, {street_count, x, y}), ...]
generations = []            # A list of populations, ultimately of size GENERATIONS
population = []             # The current population of size POPULATION_SIZE
# Represented as a list of index values that correspond to the points list
chromosome = []

FITNESS_SCORES = {}


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


def get_distance(nodeA_id, nodeB_id):
    """
    Wrapper for getting distance between two nodes
    """
    return haversine(nodeA_id, nodeB_id)


def calculate_fitness(chromosome):
    """
    Fitness is the total route cost using the haversine distance.
    The GA should attempt to minimize the fitness; minimal fitness => best fitness
    """
    # ! Points format = [(nodeId, {street_count, x, y}), ...]

    # Check if this chromosome has been calculated before,
    # Then just return the result if it had
    chromosomeTuple = tuple(chromosome)
    if(chromosomeTuple in FITNESS_SCORES):
        return [chromosome, FITNESS_SCORES[chromosomeTuple]]

    fitness = 0.0

    # Calculate the distance from the origin to the first node in the chromosome
    firstNodeId = points[chromosome[0]][0]
    fitness += get_distance(origin_id, firstNodeId)

    # Calculate the distance between each node in the chromosome
    for i in range(len(chromosome)-1):
        nodeA_id = points[chromosome[i]][0]
        nodeB_id = points[chromosome[i+1]][0]
        fitness += get_distance(nodeA_id, nodeB_id)

    # Calculate the distance from the destination to the last node in the chromosome
    lastNodeId = points[chromosome[-1]][0]
    fitness += get_distance(origin_id, lastNodeId)

    # Save the score for this chromosome to prevent recalculation in the future
    FITNESS_SCORES[chromosomeTuple] = fitness

    return [chromosome, fitness]


# initialize population
def initialize_population():
    """
    Initialize the population by creating POPULATION_SIZE chromosomes.
    Each chromosome represents the index of the point in the points list.
    Sorts the population by fitness and adds it to the generations list.
    """
    my_population = []

    # Create a new chromosome for every entry in the population
    for c in range(POPULATION_SIZE):
        # Make a new chromosome where each item in the chromosome represents an
        # index in the list of points.
        chromosomeSize = len(points)
        newChromosome = [i for i in range(chromosomeSize)]
        # Jumble up the chromosome
        RANDOM.shuffle(newChromosome)
        # Calculate the fitness
        # ! Read the ? section below to understand the [1] at the end of the line
        fitness = calculate_fitness(newChromosome)[1]
        # Make a tuple
        # FIXME: Same tuple is split then immediately reassembled
        # ? This step is a little unnecessary since calculate_fitness() already
        # ? returns a tuple in the form (chromosome, fitness), but for now, I
        # ? feel this setup is easier to read.
        pair = (newChromosome, fitness)
        # Add the chromosome to population for this generation
        my_population.append(pair)

    # Order the population by fitness
    my_population.sort(key=lambda x: x[1])

    generations.append(my_population)


def isValidChromosome(chromosome):
    # A valid chromosome will be the same length as a chromosome in the first generation.
    # A valid chromosome will contain no repeated values.
    lengthOfValidChromosome = len(generations[0][0][0])
    return (len(chromosome) == lengthOfValidChromosome) and (not checkIfDuplicates(chromosome))


def checkIfDuplicates(listOfElems):
    ''' 
    Check if given list contains any duplicates.
    Code written by Varun in his Sept. 2019 article: 
        "Python : 3 ways to check if there are duplicates in a List"

    https://thispointer.com/python-3-ways-to-check-if-there-are-duplicates-in-a-list/
    '''
    setOfElems = set()
    for elem in listOfElems:
        if elem in setOfElems:
            return True
        else:
            setOfElems.add(elem)
    return False


def repopulate(gen):
    """
    Creates a new generation by repopulation based on the previous generation.
    Calls selection, crossover, and mutate to create a child chromosome. Calculates fitness
    and continues until the population is full. Sorts the population by fitness
    and adds it to the generations list.
    """
    my_population = []
    previousPopulation = generations[gen-1]

    # Ensure you keep the top 5% of the previous generation
    retain = math.ceil(POPULATION_SIZE*0.05)
    my_population = previousPopulation[:retain]

    # Fill up the rest of the population
    while len(my_population) < POPULATION_SIZE:
        parentA, parentB = selection(previousPopulation)
        child = crossover(parentA, parentB)
        if(RANDOM.random() <= MUTATION_RATE):
            child = mutate(child)
        my_population.append(calculate_fitness(child))

    # Order the population by fitness
    my_population.sort(key=lambda x: x[1])

    generations.append(my_population)


def selection(gen):
    """
    Choose two parents and return their chromosomes
    """
    parent1, parent2 = None, None

    # Using a pretty elitest strategy the picks a chromosome from the top
    #  10%, then a 2nd chromosome from the top 70%. Note that the 2nd
    #  chromosome could potentially also come from the top 10%.
    firstParentPercentile = 0.1
    secondParentPercentile = 0.7
    firstParentIndex = RANDOM.randrange(0, round(len(gen) * firstParentPercentile))
    secondParentIndex = RANDOM.randrange(0, round(len(gen) * secondParentPercentile))

    # If the same parent is chosen, pick another
    while firstParentIndex == secondParentIndex:
        secondParentIndex = RANDOM.randrange(0, round(len(gen) * secondParentPercentile))

    # Get the parents using the parent indices
    parent1 = gen[firstParentIndex]
    parent2 = gen[secondParentIndex]

    return parent1, parent2


def crossover(p1, p2):
    """
    Strategy: Use two crossover points to directly copy a middle segment 
     from the 2nd parent. The middle segment is guranteed to start in the 
     upper half of the chromosome, but the end is only guranteed to come
     after the start. This means the segment from parent B could just be
     the first couple of genes, or at most the entirity of parent B.

     The surrounding genes, not taken from parent B, will be supplied by
     parent A. If the child already has the gene from parent B, then that
     gene will be skipped from parent A. If the child already has a gene
     in that position, the gene from parent A will be inserted at the next
     available position.

    ---------------------------------------------------------------------------

    Crossover strategy example:
    A  =  [0,1,2,3,4,5,6,7]
    B  =  [5,4,2,1,7,6,0,3]
    C_0 = [x,x,x,1,7,6,x,x] # Copy a mid section from parent B.
    C_1 = [0,x,x,1,7,6,x,x] # Get 1st gene from parent A.
    C_2 = [0,2,x,1,7,6,x,x] # Get 3rd gene from parent A because 2nd gene 
                            #   already game from parent B.
    C_3 = [0,2,3,1,7,6,x,x]
                            # Skip 4th-6th gene (1,7,6) because child already 
                            #   has genes in those positions.
    C_4 = [0,2,3,1,7,6,4,x] # Continue copying from parent A as needed.
    C_5 = [0,2,3,1,7,6,4,5] # This is the final value of child, C.
    """
    child = []

    parentAChromosome = p1[0]
    parentBChromosome = p2[0]

    # Get the size of a chromosome
    chromosomeSize = len(parentAChromosome)

    # Choose a crossover point from the first half of the chromosome
    crossOverPoint1 = RANDOM.randint(0, int(chromosomeSize/2))
    # Choose a 2nd crossover point after the first
    crossOverPoint2 = RANDOM.randint(
        crossOverPoint1 + 1, chromosomeSize)

    # *** Create Child ***
    # Make empty chromosome
    child = [None] * chromosomeSize
    # Copy genes directly from parent B
    for i in range(crossOverPoint1, crossOverPoint2):
        child[i] = parentBChromosome[i]

    # Copy remaining genes from parent A as needed
    ci = 0  # Child Index
    ai = 0  # Parent A Index
    # For every index in the child chromosome
    while ci < chromosomeSize:
        # If nothing is assigned in that gene
        if(child[ci] == None):
            # And the gene in parent A is not already in the child
            if(parentAChromosome[ai] not in child):
                # Copy the gene
                child[ci] = parentAChromosome[ai]
                # Advance the child index because a new gene was added
                ci += 1
            # Advance parent A's index because
            #  1) A gene from parent A was added.
            #  2) The gene from parent A was previously added by parent B.
            ai += 1
        # If the child already has a gene at that position from parent B,
        #  skip over that position without affecting parent A.
        else:
            ci += 1

    # Ensure that the child is composed of a valid chromosome
    if not isValidChromosome(child):
        raise RuntimeError(
            (f"Child chromosome is not valid. COP1={crossOverPoint1} "
             f"COP2={crossOverPoint2}, length={len(child)}, child={child}"))

    return child


def swap_genes(i, j, chromosome):
    """
    Swap two genes in a chromosome (swap elements in a list by index)
    """
    e = chromosome[i]
    chromosome[i] = chromosome[j]
    chromosome[j] = e
    return chromosome


def mutate(chromosome, recursiveChance=0.8, majorMutationChance=0.7):
    """
    Strategy: Randomly choose between two mutation strategies: minor and major.
      A minor mutation only swaps adjacent genes in a chromosome.
      A major mutation swaps genes anywhere within a chromosome.
      Lastly, mutations have a chance to recurse and generate more mutations.
      Return the chromosome after mutation.
    """
    mutant_child = []

    # FIXME: This code is easy to read, but VERY redudant.
    #  Come back and fix this when convient.

    # Start with an exact copy of the original chromosome
    mutant_child = chromosome.copy()

    recurse = True
    while(recurse):
        # Check to recurse
        recursiveChance = recursiveChance/2
        recurse = RANDOM.random() <= recursiveChance

        # Determine if this is a major mutation
        isMajorChange = (RANDOM.random() <= majorMutationChance)
        mutationIndex = 0
        otherIndex = 0

        if(isMajorChange):
            # Pick a random genes to mutate
            mutationIndex = RANDOM.randrange(len(chromosome))
            otherIndex = RANDOM.randrange(len(chromosome))
            # Make sure the genes aren't the same
            while mutationIndex == otherIndex:
                otherIndex = RANDOM.randrange(len(chromosome))
            # Swap the genes at the random indices inside of the chromosome
            swap_genes(mutationIndex, otherIndex, mutant_child)

        else:  # if(isMinorChange)
            # Pick a random gene to mutate
            mutationIndex = RANDOM.randrange(len(chromosome))
            # Randomly pick an adjacent gene to switch with
            otherIndex = mutationIndex + (1 if bool(RANDOM.getrandbits(1)) else -1)
            # Make sure the adjacentIndex is legal
            otherIndex = max(0, min(otherIndex, len(chromosome)-1))
            # Swap the genes at the random indices inside of the chromosome
            swap_genes(mutationIndex, otherIndex, mutant_child)

        # Ensure that the mutated chromosome is valid
        if not isValidChromosome(mutant_child):
            raise RuntimeError(
                (f"Mutant chromosome is not valid. isMajorMutation={isMajorChange}, "
                    f"MutationIndex={mutationIndex}, OtherIndex={otherIndex}, "
                    f"recurisveChance={recursiveChance}, mutant={mutant_child}"))

    return mutant_child


def uniq(lst):
    '''
        Removes duplicates from a sorted list.
        Code written by Stack Overflow user user4815162342
        in a response to the Nov. 2012 Question:
            "TypeError: unhashable type: 'list' when using built-in set function"

        https://stackoverflow.com/questions/13464152/typeerror-unhashable-type-list-when-using-built-in-set-function
    '''
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item


def get_homogeneity(gen):
    '''
    Calculate the level of homogeneity for a given generation.
    Aka, The fraction of the population that is not unique.
    '''
    pop = generations[gen]
    popSet = list(uniq(pop))
    ucp = len(popSet)  # Number of unique creatures in the population
    homogeneity = ucp / POPULATION_SIZE
    homogeneity = 1 - homogeneity
    return homogeneity


def run_ga():
    """
    Initialize and repopulate until you have reached the maximum generations
    """
    initialize_population()

    # Use these variables to keep track of the lowest fitness found so far
    lowestFitness = 1000000
    lowestFitnessGen = -1
    # Use this variable to track if it is necessary to display a new graph
    lowestFitnessPrevGraph = 110000
    # Use these variables to determine how often an update should be displayed
    # ? I found it useful to keep these seperated, as I often wanted detailed
    # ?  written info, but didn't necessarily need to see the graph. This
    # ?  allowed me to strike a balance between extra information and
    # ?  and wasted cycles displaying an unnecessary graph.
    graphDisplayRate = DISPLAY_RATE
    textDisplayRate = int(DISPLAY_RATE / 10)

    # For every generation
    for gen in range(GENERATIONS - 1):  # Note, you already ran generation 1

        # Create a new generation based on the previous generation
        # Note that +1 needs to be added because generation 0 is the initialized generation.
        # FIXME: Couldn't I just update the for loop to use range(0, ...) to remove the +1 here?
        repopulate(gen + 1)

        # Determine the best fitness of this generation
        currentFitness = generations[gen][0][1]
        # Update the best fitness found so far if necessary
        if(currentFitness < lowestFitness):
            lowestFitness = currentFitness
            lowestFitnessGen = gen

        # Display text update
        if gen % textDisplayRate == 0:
            # Calculate the current homogenity level
            homogeneity = get_homogeneity(gen) * 100  # Multiply by 100 to convert to a percentage
            # Display the information update
            print((
                f"Generation Stuff: (Gen #: {gen}, Fitness: {currentFitness}, "
                f"homogeneity: {homogeneity:.2f}%, "
                f"Best: ({lowestFitness}, #{lowestFitnessGen}))"))

        # Display graph update
        # Only update if the latest graph is different from the previous generation's graph.
        if gen % graphDisplayRate == 0 and lowestFitness < lowestFitnessPrevGraph:
            lowestFitnessPrevGraph = lowestFitness
            show_route(gen)


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
    #print("The fitness for generation", generation_number, "was", the_fitness)
    print(f"Latest Top Chromosome: {the_route}")


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

    print(f"Random seed is: {RANDOM_SEED}")
    print(f"First 3 values are: {RANDOM.randrange(0,10)}, {RANDOM.randrange(0,10)}, {RANDOM.randrange(0,10)}")

    start_time = time.time()
    run_ga()
    # show_route(0)
    # show_route(math.floor(GENERATIONS/2))
    show_route(GENERATIONS-1)

    plot_ga()

    end_time = time.time()
    run_time = end_time - start_time
    minutes = math.floor(run_time / 60)
    seconds = run_time % 60

    print(f"Run time: {minutes}min {seconds}sec")


main()
