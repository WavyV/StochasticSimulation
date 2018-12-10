import numpy as np
import math
import random

# Get the current tour score by summing the distance matrix values for the current tour
def getTourScore(tour):
    total = 0
    for i in range(len(tour)-1):
        total += distM[tour[i]][tour[i+1]]
    total += distM[tour[-1]][tour[0]]
    return total

# Propose a new tour by randomly selecting 2 cities and reversing the values between
def proposeNewTour(tour):
    rand1 = random.randint(0, len(tour)-1)
    rand2 = random.randint(rand1+1, len(tour)) # rand1 < rand2
    tour[rand1:rand2] = reversed(tour[rand1:rand2])
    return tour

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

def markovChain(tour, markovChainLength, temperature):
    curTour = tour
    curScore = getTourScore(curTour)
    bestTour, bestScore = curTour, curScore
    accepted = 0
    for _ in range(markovChainLength):
        newTour = proposeNewTour(curTour)
        newScore = getTourScore(newTour)
        A = min(1, math.exp(-(newScore - curScore)/temperature))
        if(A == 1): #Accept new tour
            curTour, curScore = newTour, newScore
            accepted += 1
            if(curScore < bestScore):
                bestScore, bestTour = curScore, curTour
        else:
            if(A < random.random()):
                curTour, curScore = newTour, newScore
                accepted += 1

    return curTour, curScore, accepted/markovChainLength

distM = np.load('distM/distMeil51.npy')
# tour = [i for i in range(51)]
# random.shuffle(tour) # take a random starting point
tour = greedyTour(51)
opt = [1,22,8,26,31,28,3,36,35,20,2,29,21,16,50,34,30,9,49,10,39,33,45,15,44,42,
    40,19,41,13,25,14,24,43,7,23,48,6,27,51,46,12,47,18,4,17,37,5,38,11,32]
opt = [opt[i]-1 for i in range(len(opt))]
# print(opt, len(opt))
# print(tour, len(tour))

# print(getTourScore(opt))
# print(opt)

# parameters
temperature = 100
stopTemperature = 0.01
stopIteration = 10000
alpha = 0.995
delta = 0.1

curScore = getTourScore(tour)
curTour = tour
scores = [curScore]
bestScore = curScore
bestTour = curTour
iteration = 0
while iteration < stopIteration and temperature > stopTemperature:
    curTour, curScore, p = markovChain(curTour, 200, temperature)
    if(curScore < bestScore):
        bestTour, bestScore = curTour, curScore
    temperature *= alpha

    if(iteration % 100 == 0):
        print(curScore, temperature)
    iteration += 1

# print(temperature, iteration)
# print(accepted/iteration)
print(bestTour)
print(bestScore)
