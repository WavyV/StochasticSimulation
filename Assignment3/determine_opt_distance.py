import numpy as np
import math

D = 280

lines = [line.rstrip() for line in open('TSP-Configurations/a280.opt.tour.txt', 'r')]
datalines = lines[5:D+5]

distance = 0
distM = np.load('distM/distMa280.npy')

for i in range(D-1):
    x = datalines[i]
    y = datalines[i+1]
    distance += distM[int(x)-1][int(y)-1]

x = datalines[D-1]
y = datalines[0]
distance += distM[int(x)-1][int(y)-1]

print(distance)
