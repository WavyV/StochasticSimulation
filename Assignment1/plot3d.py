# Import packages
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from numba import jit
from tqdm import tqdm
import numpy as np
import pandas as pd
import math
import itertools
import random

MAXITER = 300
CMAP = 'gnuplot2'
DPI = 300

# Functions needed to compute the Mandelbrot set (adapted from https://gist.github.com/jfpuget/60e07a82dece69b011bb)
@jit
def mandelbrot(c, maxiter):
    z = c
    for n in range(maxiter):
        if abs(z) > 3:
            return n
        z = z*z + c
    return 0

@jit
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, maxiter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width, height))
    for i in range(width):
        for j in range(height):
            n3[i,j] = mandelbrot(r1[i] + 1j*r2[j], maxiter)
    return (r1, r2, n3)

def mandelbrot_image(xmin, xmax, ymin, ymax, width, height, maxiter, cmap):
    img_width = DPI * width
    img_height = DPI * height
    x,y,z = mandelbrot_set(xmin ,xmax, ymin, ymax, img_width, img_height, maxiter)

    fig, ax = plt.subplots(figsize=(width, height), dpi=DPI)
    ticks = np.arange(0, img_width+1, int(img_width/4))
    x_ticks = xmin + (xmax-xmin)*ticks/img_width
    plt.xticks(ticks, x_ticks)
    y_ticks = ymin + (ymax-ymin)*ticks/img_width
    plt.yticks(ticks, y_ticks)

    norm = colors.PowerNorm(0.5)
    ax.imshow(z.T, cmap=cmap, origin='lower', norm=norm)
    plt.show()

# Basic hit and miss algorithm
def basic_hit_miss(shots):
    hits = 0
    for s in range(shots):
        x_random = random.randint(0, img_width-1)
        y_random = random.randint(0, img_height-1)
        if(z[x_random, y_random] == 0):
            hits += 1
    estimated_area = (hits / shots) * total_area
#     print(estimated_area)
    error = abs(estimated_area - mandelbrot_area)
#     print(error)
    return estimated_area, error

# Estimation of area of mandelbrot set using selected parameter settings
img_width = DPI * 5
img_height = DPI * 5
xmin, xmax, ymin, ymax = -2, 1, -1.5, 1.5
x,y,z = mandelbrot_set(xmin, xmax, ymin, ymax, img_width, img_height, maxiter = MAXITER)
mandelbrot_pixels = np.sum(z == 0)
mandelbrot_ratio = mandelbrot_pixels / (img_width * img_height)
total_area = abs(xmin - xmax) * abs(ymin - ymax)
mandelbrot_area = mandelbrot_ratio * total_area

# x = [100, 1000, 2000, 5000, 10000, 50000, 100000]
x = [2000*i for i in range(1, 51)]
print(x)
# s = [100, 300, 500, 700, 900]
s = [200, 400, 600, 800]
grid = np.meshgrid(x,s)

reps = 10
totals = np.zeros((len(s), len(x)))
for i in tqdm(range(reps)):
    for si in range(len(s)):
#         MAXITER = s[si]
#         DPI = s[si]
#         result = []
        for j in range(len(x)):
            estimated_area, error = basic_hit_miss(x[j])
            totals[si][j] += estimated_area

totals = totals/reps
print(totals)
# totals = np.array(totals)
# mean, std = [], []
# for i in range(len(x)):
#     mean.append(np.mean(totals[:, i]))
#     std.append(np.std(totals[:, i]))
# plt.figure()
# plt.plot(x, mean, color="blue")
# plt.yscale('log')
# plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(grid[0], grid[1], totals, linewidth=0, antialiased=False)

ax.view_init(10, 200)
ax.set_xlabel('Samples')
ax.set_ylabel('Iterations')
ax.set_zlabel('Estimated area')
plt.show()
