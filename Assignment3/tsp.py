import numpy as np
import math

# lines = [line.rstrip() for line in open('TSP-Configurations/eil51.tsp.txt', 'r')]
# datalines = lines[6:57]
# print(datalines)
#
# distanceM = np.zeros((51,51), dtype='float32')
# for i in range(51):
#     coord1 = datalines[i].split()
#     x1, y1 = int(coord1[1]), int(coord1[2])
#     for j in range(51):
#         coord2 = datalines[j].split()
#         x2, y2 = int(coord2[1]), int(coord2[2])
#         print(x1, x2)
#         distanceM[i][j] = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
#
# np.save('distM/distMeil51.npy', distanceM)
distM = np.load('distM/distMeil51.npy')
print(distM)
