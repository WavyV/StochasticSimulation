import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


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

        T *= alpha
        iteration += 1

    return(bestScore)


# Load the distance matrix of the appropriate problem
problem = 1  #1: eil51, 2: a280, 3: pcb442
if problem == 1:
    distM = np.load('distM/distMeil51.npy')
    D = 51
elif problem == 2:
    distM = np.load('distM/distMa280.npy')
    D = 280
elif problem == 3:
    distM = np.load('distM/distMpcb442.npy')
    D = 442

# parameters
T0 = 90
coolingSchedule = 'notImplementedError()'
greedy = 0
runs = 1

alphas = [0.8, 0.85, 0.9, 0.925, 0.95, 0.98, 0.99, 0.999, 0.999, 0.9999]
maxiters = [500, 1000, 2500, 5000, 10000, 25000]

i = 0
means = np.zeros((len(alphas), len(maxiters)))
stds = np.zeros((len(alphas), len(maxiters)))
for alpha in tqdm(alphas):
    j = 0
    for maxiter in maxiters:
        results = []
        for _ in range(runs):
            results.append(tsp(distM, D, greedy, T0, alpha, maxiter))
        means[i, j] = np.mean(results)
        stds[i, j] = 1.96*np.std(results)/np.sqrt(runs)
        j += 1
    i += 1

print(means)
print(stds)

normalized_means = means / np.max(means)
width = 4
height = 4
DPI = 300
img_width = DPI * width
img_height = DPI * height
fig, ax = plt.subplots(figsize=(width, height), dpi=DPI)
ticks = np.arange(0, img_width)
x_ticks = [0.5, 1, 2.5, 5, 10, 25]
plt.xticks(ticks, x_ticks)
y_ticks = alphas
plt.yticks(ticks, y_ticks)
plt.imshow(means, origin='lower')
plt.xlabel('Maximum Number of Iterations (x10^3)')
plt.ylabel('Alpha')
plt.colorbar()
plt.show()
