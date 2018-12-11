import numpy as np
import math

D = 442

lines = [line.rstrip() for line in open('TSP-Configurations/pcb442.tsp.txt', 'r')]
datalines = lines[6:D+6]

distanceM = np.zeros((D,D), dtype='float32')
points = []
for i in range(D):
    coord1 = datalines[i].split()
    x1, y1 = float(coord1[1]), float(coord1[2])
    x1, y1 = int(x1), int(y1)
    points.append((x1, y1))
    for j in range(D):
        coord2 = datalines[j].split()
        x2, y2 = float(coord2[1]), float(coord2[2])
        x2, y2 = int(x2), int(y2)
        distanceM[i][j] = round(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))

np.save('distM/distMpcb442.npy', distanceM)
np.save('distM/pcb442_coord.npy', points)
