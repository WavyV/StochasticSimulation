# EVOLUTIONARY ALGORITHM TSP

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Get the current tour score by summing the distance matrix values for the current tour
def getTourScore(tour):
    total = 0
    for i in range(len(tour)-1):
        total += distM[tour[i]-1][tour[i+1]-1]
    total += distM[tour[-1]-1][tour[0]-1]
    return total


# Propose a new tour by randomly selecting 2 cities and reversing the values between
def proposeNewTour(tour):
    newTour = tour
    rand1 = random.randint(0, len(tour)-1)
    rand2 = random.randint(rand1+1, len(tour)) # rand1 < rand2
    stuff = []
    for i in range(rand2-1, rand1-1, -1):
        stuff.append(tour[i])
    newTour[rand1:rand2] = stuff
    return newTour


# Rank the population on fitness
def rankPopulation(population):
    ranked_population = np.zeros((len(population), D+1))
    order = np.argsort(population[:, D])
    for i in range(len(population)):
        ranked_population[i, :] = population[order[i], :]
    return ranked_population


def greedyTour(size):
    tour = [random.randint(1, size)]
    for _ in range(size-1):
        minIndex, minValue = 0, 100000
        for i in range(1, size+1):
            if tour[-1] == i or i in tour:
                continue
            elif distM[tour[-1]-1][i-1] < minValue:
                minIndex, minValue = i, distM[tour[-1]-1][i-1]
        tour.append(minIndex)

    return tour


#distM = np.load('distM/distMeil51.npy')
distM = np.load('distM/distMa280.npy')
#distM = np.load('distM/distMpcb442.npy')


N = 100  #Number of individuals
D = 280  #Number of cities (51, 280 or 442)
maxiter = 7500
INIT = 0 # select 1 for greedy, select 0 for totally random
best = np.zeros((maxiter))

# Initialisation of population and initial fitness calculation
# Each individual solution is a row where
# the first D indicies the route and
# the last index contains the tour score ~ fitness
population = np.zeros((N, D+1))
for n in range(N):
    if INIT == 0:
        population[n, 0:D] = np.arange(1, D+1)
        np.random.shuffle(population[n, 0:D])
        tour = population[n, 0:D]
        population[n, D] = getTourScore(tour.astype('int'))
    elif INIT == 1:
        population[n, 0:D] = greedyTour(D)
        tour = population[n, 0:D]
        population[n, D] = getTourScore(tour.astype('int'))

population = rankPopulation(population)

iter = 0
for iter in tqdm(range(maxiter)):

    # Each individual creates 1 offspring by having part of the sequence inverted
    children = np.zeros((N, D+1))
    for n in range(N):
        tour = population[n, 0:D]
        tour = proposeNewTour(tour.astype('int'))
        children[n, 0:D] = tour
        children[n, D] = getTourScore(tour.astype('int'))

    # Merge the parent and children populations
    merged_population = np.zeros((2*N, D+1))
    merged_population[:N, :] = population
    merged_population[N:, :] = children

    # And pick the best 100 to form the next parent population
    ranked_total = rankPopulation(merged_population)
    population = ranked_total[:N, :]
    iter += 1
    best[iter-1] = population[0, D]

np.save('results/evol_res.npy', best)
print(population[:5, :])
plt.plot(np.arange(maxiter), best)
plt.xlabel('Iteration')
plt.ylabel('Tour Score')
plt.title('Tour Score of Best Individual')
plt.show()
