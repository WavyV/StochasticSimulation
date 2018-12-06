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

distM = np.load('distM/distMeil51.npy')
tour = [i for i in range(51)]
random.shuffle(tour) # take a random starting point

opt = [1,22,8,26,31,28,3,36,35,20,2,29,21,16,50,34,30,9,49,10,39,33,45,15,44,42,
    40,19,41,13,25,14,24,43,7,23,48,6,27,51,46,12,47,18,4,17,37,5,38,11,32]
opt = [opt[i]-1 for i in range(len(opt))]
print(getTourScore(opt))
print(opt)
# parameters
temperature = 1000
stopTemperature = 0.00000001
stopIteration = 10000000
alpha = 0.9995

curScore = getTourScore(tour)
curTour = tour
bestScore = curScore
bestTour = curTour
accepted = 0
iteration = 0
while iteration < stopIteration and temperature > stopTemperature:
    # print(iteration, temperature)
    newTour = proposeNewTour(curTour)
    newScore = getTourScore(newTour)
    if(newScore < curScore): #Accept new tour
        curTour = newTour
        curScore = newScore
        accepted += 1
        if(curScore < bestScore):
            bestScore = curScore
            bestTour = curTour
    else:
        uni = random.uniform(0,1)
        bolz = math.exp(-abs(newScore - curScore)/temperature)
        if(bolz < uni):
            curTour = newTour
            curScore = newScore
            accepted += 1

    temperature = alpha*temperature
    iteration += 1

print(temperature, iteration)
print(accepted/iteration)
print(bestTour)
print(bestScore)
