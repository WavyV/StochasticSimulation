import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


# Get the current tour score by summing the distance matrix values for the current tour
def getTourScore(tour):
    total = 0
    # print(tour)
    for i in range(len(tour)-1):
        total += distM[tour[i]][tour[i+1]]
    total += distM[tour[-1]][tour[0]]
    # print(" XXXXX")
    return total


def proposeNewTour(route):
    c1 = np.random.choice(range(0, len(route)-2))
    c2 = np.random.choice(range(c1+2, len(route)))
    new_route = np.append(route[0:c1], np.flip(route[c1:c2+1],0))
    new_route = np.append(new_route, route[c2+1:])
    return new_route

distM = np.load('distM/distMa280.npy')
tour = [i for i in range(280)]
random.shuffle(tour) # take a random starting point

bestScore = getTourScore(tour)
bestTour = tour
scores = []
notImproved = 0
maxiter = 750000
for _ in tqdm(range(maxiter)):
    newTour = proposeNewTour(bestTour)
    newScore = getTourScore(newTour)
    if newScore < bestScore:
        notImproved = 0
        bestScore, bestTour = newScore, newTour
    else:
        notImproved += 1

    if notImproved > 5000:
        x = [i for i in range(280)]
        random.shuffle(x)
        bestTour = x
        bestScore = getTourScore(bestTour)
        notImproved = 0

    scores.append(bestScore)

np.save('results/greedy.npy', scores)
print(min(scores))
plt.figure()
plt.plot(np.arange(maxiter), scores)
plt.show()
