import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def getNextTemperature(coolingSchedule, iteration, curTemp, T0, TN, maxiter):
    if coolingSchedule == "linear":
        return T0 - iteration*(T0 - TN)/float(maxiter)
    elif coolingSchedule == "exponential":
        return T0*(TN/T0)**(iteration/float(maxiter))
    elif coolingSchedule == 'sigmoid':
        return  TN + (T0 - TN)*(1./(1.+math.exp(((2.0*math.log(T0-TN))/maxiter)*(iteration-0.5*maxiter))))
    elif coolingSchedule == 'other':
        return T0 - iteration**(math.log(T0-TN)/math.log(maxiter))

def tsp(distanceMatrix, D, T0, coolingSchedule, maxiter, minTemp):
    T = T0

    tour = np.arange(0, D)
    np.random.shuffle(tour)

    curScore = getTourScore(tour, distanceMatrix)
    curTour = tour
    bestScore = curScore
    bestTour = curTour
    iteration = 0
    temp, iter, scores = [], [], []
    while iteration < maxiter:
        newTour = proposeNewTour(curTour)
        newScore = getTourScore(newTour, distanceMatrix)
        if(newScore < curScore): #Accept new tour if proposed tour better
            notImproved = 0
            curTour, curScore = newTour, newScore
        elif(random.random() < math.exp(-abs(newScore - curScore) / T)): #Otherwise a small chance we'll accept it anyway
            curTour, curScore = newTour, newScore

        T = getNextTemperature(coolingSchedule, iteration, T, T0, minTemp, maxiter)
        iteration += 1
        scores.append(curScore)

    return scores

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

T0 = 379
reps = 10
maxiter = 1000000
coolingSchedules = ['linear', 'exponential', 'sigmoid']
means = [np.zeros((reps, maxiter)) for _ in range(len(coolingSchedules))]
for i in tqdm(range(len(coolingSchedules))):
    for j in range(reps):
        scores = tsp(distM, D, T0, coolingSchedules[i], maxiter, 0.05)
        means[i][j] = scores

x = np.arange(maxiter)
plt.figure(figsize=(8,6))
total = [np.zeros(maxiter) for _ in range(len(coolingSchedules))]
std = [np.zeros(maxiter) for _ in range(len(coolingSchedules))]
colors = ['blue', 'orange', 'green']
for i in range(len(coolingSchedules)):
    for j in range(maxiter):
        total[i][j] = sum(means[i][:,j])/float(reps)
        std[i][j] = 1.96*np.std(means[i][:,j])/np.sqrt(reps)
    plt.plot(x, total[i], label=coolingSchedules[i], color=colors[i])
    plt.fill_between(x, total[i], total[i]-std[i], color=colors[i], alpha=0.5)
    plt.fill_between(x, total[i], total[i]+std[i], color=colors[i], alpha=0.5)
plt.plot(x, [2579 for _ in range(maxiter)], linestyle="--", color="grey")
plt.xlabel('iterations')
plt.ylabel('Route length')
plt.title('Different cooling schedules')
plt.legend()
plt.savefig("diff_cooling_sched_res_280.png")
plt.show()
