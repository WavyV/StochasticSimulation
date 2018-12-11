import numpy as np
import math
import random


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


#Calculate acceptance probability given a set of positive
#transitions and a temperature
def calculateXi(neighbours, random_starts, Tn):
    x, y = 0, 0
    for i in range(len(neighbours)):
        y += np.exp(-neighbours[i]/Tn)
        x += np.exp(-random_starts[i]/Tn)
    return(np.log(x/y))


distM = np.load('distM/distMeil51.npy')

#Find some positive initial transitions
N = 10000
D = 51
random_starts = np.zeros((N))
neighbours = np.zeros((N))
for i in range(N):
    accepted = False
    while accepted == False:
        tour = np.arange(D)
        np.random.shuffle(tour)
        x1 = getTourScore(tour)
        newTour = proposeNewTour(tour)
        x2 = getTourScore(newTour)
        if(x2 - x1 < 0):
            random_starts[i] = x1
            neighbours[i] = x2
            accepted = True

#Converge to the optimal starting temperature
Tn = 5
eps = 0.001
delta = 1
p = 2
while delta > eps:
    t_prev = Tn
    xiTn = calculateXi(neighbours, random_starts, Tn)
    Tn = Tn * (xiTn/np.log(0.8))**(1/p)
    delta = np.abs(np.exp(xiTn) - 0.8)
    print(Tn)
