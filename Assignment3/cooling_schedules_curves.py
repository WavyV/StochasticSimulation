import numpy as np
import math
import random
import matplotlib.pyplot as plt

# Get the current tour score by summing the distance matrix values for the current tour
def getTourScore(tour, distM):
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

def getNextTemperature(coolingSchedule, iteration, T0, TN, maxiter):
    if coolingSchedule == "linear":
        return T0 - iteration*(T0 - TN)/maxiter
    elif coolingSchedule == "exponential":
        return T0*(TN/T0)**(iteration/float(maxiter))
    elif coolingSchedule == 'sigmoid':
        return  TN + (T0 - TN)*(1./(1.+math.exp(((2.0*math.log(T0-TN))/maxiter)*(iteration-0.5*maxiter))))
    elif coolingSchedule == 'logarithmic':
        return T0 - iteration**(math.log(T0-TN)/math.log(maxiter))
        # return (T0-TN)*(maxiter+1)/maxiter/(iteration+1)+T0-(T0-TN)*(maxiter+1)/maxiter

def tsp(distanceMatrix, T0, coolingSchedule, maxiter, minTemp):
    T = T0

    tour = np.arange(0, 280)
    np.random.shuffle(tour)

    curScore = getTourScore(tour, distanceMatrix)
    curTour = tour
    bestScore = curScore
    bestTour = curTour
    iteration = 0
    temp, iter = [], []
    while iteration < maxiter:
        # newTour = proposeNewTour(curTour)
        # newScore = getTourScore(newTour, distanceMatrix)
        # if(newScore < curScore): #Accept new tour if proposed tour better
        #     notImproved = 0
        #     curTour, curScore = newTour, newScore
        #     if(curScore < bestScore):
        #         bestScore, bestTour = curScore, curTour
        # elif(random.random() < math.exp(-abs(newScore - curScore) / T)): #Otherwise a small chance we'll accept it anyway
        #     curTour, curScore = newTour, newScore

        T = getNextTemperature(coolingSchedule, iteration, T0, minTemp, maxiter)
        iteration += 1
        temp.append(T)
        iter.append(iteration)

    return temp, iter

problem = 2  #1: eil51, 2: a280, 3: pcb442
if problem == 1:
    distM = np.load('distM/distMeil51.npy')
    D = 51
elif problem == 2:
    distM = np.load('distM/distMa280.npy')
    D = 280
elif problem == 3:
    distM = np.load('distM/distMpcb442.npy')
    D = 442

T0 = 379.
reps = 1
maxiter = 1000000
coolingSchedules = ['linear', 'exponential', 'sigmoid']
# means = [np.zeros((reps, maxiter)) for _ in range(len(coolingSchedules))]
plt.figure(figsize=(8,4))
for i in range(len(coolingSchedules)):
    for j in range(reps):
        temp, iter = tsp(distM, T0, coolingSchedules[i], maxiter, 0.05)
        plt.plot(iter, temp, label=coolingSchedules[i])

plt.xlabel('iterations')
plt.ylabel('temperature')
plt.title('Different cooling schedules')
plt.legend()
plt.savefig("diff_cooling_sched2.png")
plt.show()
