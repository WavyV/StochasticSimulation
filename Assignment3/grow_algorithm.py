import numpy as np
import math
import random
from tqdm import tqdm

def greedyTour(size):
    tour = [random.randint(1, size)]
    for j in range(size-1):
        minIndex, minValue = 0, 100000
        for i in range(1, size+1):
            if tour[-1] == i or i in tour:
                continue
            elif distM[tour[-1]-1][i-1] < minValue:
                minIndex, minValue = i, distM[tour[-1]-1][i-1]
        tour.append(minIndex)

    return tour

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


def getTourScore(tour):
    total = 0
    for i in range(len(tour)-1):
        total += distM[tour[i]][tour[i+1]]
    total += distM[tour[-1]][tour[0]]
    return total



#distM = np.load('distM/distMeil51.npy')
#distM = np.load('distM/distMa280.npy')
distM = np.load('distM/distMpcb442.npy')
#cities = load("TSP-Configurations/eil51.tsp.txt")
#cities = load("TSP-Configurations/a280.tsp.txt")
cities = load("TSP-Configurations/pcb442.tsp.txt")

N = 1000
results = np.zeros((N))
for n in tqdm(range(N)):
    results[n] = getTourScore(np.asarray(greedyTour(442))-1)

print(np.mean(results))
print(1.96*np.std(results, ddof=1)/np.sqrt(N))
print(np.min(results))
