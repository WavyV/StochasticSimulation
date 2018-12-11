import numpy as np
import math
import random
import matplotlib.pyplot as plt

# Get the current tour score by summing the distance matrix values for the current tour
def getTourScore(tour):
    total = 0
    for i in range(len(tour)-1):
        total += distM[tour[i]][tour[i+1]]
    total += distM[tour[-1]][tour[0]]
    return total


def proposeNewTour(route):
    c1 = np.random.choice(range(0, len(route)-2))
    c2 = np.random.choice(range(c1+2, len(route)))
    new_route = np.append(route[0:c1], np.flip(route[c1:c2+1],0))
    new_route = np.append(new_route, route[c2+1:])
    return new_route


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

distM = np.load('distM/distMeil51.npy')
# tour = [i for i in range(51)]
# random.shuffle(tour) # take a random starting point
tour = greedyTour(51)
#print(tour)
# opt = [1,22,8,26,31,28,3,36,35,20,2,29,21,16,50,34,30,9,49,10,39,33,45,15,44,42,
#     40,19,41,13,25,14,24,43,7,23,48,6,27,51,46,12,47,18,4,17,37,5,38,11,32]
# opt = [opt[i]-1 for i in range(len(opt))]
# print(getTourScore(opt))
# print(opt)

# parameters
T0 = 90
temperature = T0
stopTemperature = 0.05
stopIteration = 25000
alpha = 0.9997
delta = 0.1

curScore = getTourScore(tour)
curTour = tour
scores = [curScore]
bestScore = curScore
bestTour = curTour
accepted = 0
iteration = 0
notImproved = 0
iter, temperatures = [], []
while iteration < stopIteration:
    # print(iteration, temperature)
    newTour = proposeNewTour(curTour)
    newScore = getTourScore(newTour)
    if(newScore < curScore): #Accept new tour
        notImproved = 0
        curTour, curScore = newTour, newScore
        accepted += 1
        if(curScore < bestScore):
            bestScore, bestTour = curScore, curTour
    else:
        if(random.random() < math.exp(-abs(newScore - curScore)/temperature)):
            curTour, curScore = newTour, newScore
            accepted += 1
        else:
            notImproved += 1

    # if(iteration % 100 == 0):
    #     print(curScore, temperature)

    # temperature = temperature*alpha
    print(temperature, iteration)
    # temperature = (T0 - stopTemperature)/(1+math.exp(0.3*(iteration-stopIteration/2))) + stopTemperature
    # temperature = stopTemperature + (T0 - stopTemperature)*(1./(1.+math.exp(((2.0*math.log(T0-stopTemperature))/stopIteration)*(iteration-0.5*stopIteration))))
    # temperature = T0 - iteration**(math.log(T0-stopTemperature)/math.log(stopIteration))
    temperature = T0/(1+178*math.log(1+iteration))
    temperatures.append(temperature)
    iter.append(iteration)
    # if(notImproved < 50):
    #     temperature *= alpha
    # else:
    #     temperature *= 1./alpha
    # temperature = temperature*(1+math.log(1+delta)*temperature/3*np.std(scores))**-1
    iteration += 1

# print(temperature, iteration)
print(accepted/iteration)
print(bestTour)
print(bestScore)
print(temperature)

plt.figure()
plt.plot(iter, temperatures)
# plt.yscale('log')
plt.show()
