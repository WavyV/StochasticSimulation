import numpy as np
import matplotlib.pyplot as plt
import math
import random
import csv
import copy

## load files into python
def load(filename):
    data = []
    with open(filename, 'r') as file:
        for row in file:
            try:
                if int(row.split()[0]) > 0:
                    data.append(row.split())
            except (RuntimeError, TypeError, NameError, ValueError):
                pass
    data = np.asarray(data).astype('float64')
    return data

# create map with cities and the selected route (input = cities coordinates and route)
def map_route(D, route, iter):
    if D == 51:
        cities = load("TSP-Configurations/eil51.tsp.txt")
    elif D == 280:
        cities = load("TSP-Configurations/a280.tsp.txt")
    elif D == 442:
        cities = load("TSP-Configurations/a280.tsp.txt")

    x, y = cities[:,1], cities[:,2]
    xlist, ylist = [], []
    for i in route:
        xlist.append(cities[i,1])
        ylist.append(cities[i,2])

    xlist.append(cities[route[0],1])
    ylist.append(cities[route[0],2])

    plt.plot(x,y, 'ro')
    plt.plot(xlist, ylist, 'g-')
    plt.xlabel('X')
    plt.ylabel('Y')
    if isinstance(iter, int):
        plt.title('Route at Iteration %d' % iter)
    else:
        plt.title('Optimal Route')
    plt.show()

    return


# Get the current tour score by summing the distance matrix values for the current tour
def getTourScore(tour):
    total = 0
    for i in range(len(tour)-1):
        total += distM[tour[i]][tour[i+1]]
    total += distM[tour[-1]][tour[0]]
    return total


# Take a tour and reverse some nodes within a random interval and propose it as a new tour
def proposeNewTour(route):
    c1 = np.random.choice(range(0, len(route)-2))
    c2 = np.random.choice(range(c1+2, len(route)))
    new_route = np.append(route[0:c1], np.flip(route[c1:c2+1],0))
    new_route = np.append(new_route, route[c2+1:])
    return new_route


# Find a tour using the "greedy" approach
def greedyTour(size):
    tour = [random.randint(0, size-1)]
    for _ in range(size-1):
        minIndex, minValue = 0, 100000
        for i in range(size):
            if tour[-1] == i or i in tour:
                continue
            elif distM[tour[-1]][i] < minValue:
                minIndex, minValue = i, distM[tour[-1]][i]
        tour.append(minIndex)
    return tour


def tsp(distanceMatrix, D, greedy, T0, coolingSchedule, maxiter):

    alpha = coolingSchedule
    T = T0
    # Start with a greedy solution as initial?
    greedy = 0  #0: start with completely random tour, 1: start with greedy solution
    if greedy == 0:
        tour = np.arange(0, D)
        np.random.shuffle(tour)
    elif greedy == 1:
        tour = greedyTour(D)

    curScore = getTourScore(tour)
    curTour = tour
    scores = [curScore]
    bestScore = curScore
    bestTour = curTour
    iteration = 0
    while iteration < maxiter:
        newTour = proposeNewTour(curTour)
        newScore = getTourScore(newTour)
        if(newScore < curScore): #Accept new tour if proposed tour better
            notImproved = 0
            curTour, curScore = newTour, newScore
            if(curScore < bestScore):
                bestScore, bestTour = curScore, curTour
        elif(random.random() < math.exp(-abs(newScore - curScore) / T)): #Otherwise a small chance we'll accept it anyway
            curTour, curScore = newTour, newScore

        if iteration % (maxiter/4) == 0:
            map_route(D, curTour, iteration)
        elif iteration == maxiter - 1:
            map_route(D, curTour, iteration+1)

        T *= alpha
        iteration += 1

    return(bestScore)


# Load the distance matrix of the appropriate problem
problem = 1  #1: eil51, 2: a280, 3: pcb442
if problem == 1:
    distM = np.load('distM/distMeil51.npy')
    opt_route = load("TSP-Configurations/eil51.opt.tour.txt")
    opt_route = np.asarray(opt_route).astype(int) - 1
    D = 51
elif problem == 2:
    distM = np.load('distM/distMa280.npy')
    opt_route = load("TSP-Configurations/a280.opt.tour.txt")
    opt_route = np.asarray(opt_route).astype(int) - 1
    D = 280
elif problem == 3:
    distM = np.load('distM/distMpcb442.npy')
    opt_route = load("TSP-Configurations/pcb442.opt.tour.txt")
    opt_route = np.asarray(opt_route).astype(int) - 1
    D = 442

# parameters
T0 = 90
coolingSchedule = 'notImplementedError()'
greedy = 0
runs = 1
alpha = 0.9995
maxiter = 25000

print(tsp(distM, D, greedy, T0, alpha, maxiter))

map_route(D, opt_route, 'optimal')
