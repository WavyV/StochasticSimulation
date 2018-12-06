import numpy as np
import math

lines = [line.rstrip() for line in open('TSP-Configurations/eil51.tsp.txt', 'r')]
datalines = lines[6:57]

distanceM = np.zeros((51,51), dtype='float32')
points = []
for i in range(51):
    coord1 = datalines[i].split()
    x1, y1 = int(coord1[1]), int(coord1[2])
    points.append((x1, y1))
    for j in range(51):
        coord2 = datalines[j].split()
        x2, y2 = int(coord2[1]), int(coord2[2])
        distanceM[i][j] = round(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))

np.save('distM/distMeil51.npy', distanceM)
np.save('distM/eil51_coord.npy', points)
