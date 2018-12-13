import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

evol = np.load('results/evol_res.npy')
SA = np.load('results/SA_res.npy')
greedy = np.load('results/greedy.npy')

print(min(evol))
print(min(SA))
print(min(greedy))
plt.figure(figsize=(8,6))
plt.plot([i*100 for i in range(7500)], evol, label="Evolutianary")
plt.plot(np.arange(750000), SA, label="Simulated annealing")
plt.plot(np.arange(750000), greedy, label="Greedy")
plt.plot(np.arange(750000), [2579 for _ in range(750000)], linestyle="--", color="grey", label="Optimal")
plt.xlabel("Iterations")
plt.ylabel("Route length")
plt.title("Performance comparison of different methods", fontsize=14)
plt.yscale('log')
plt.legend()
plt.savefig("methods_vs_log.png")
plt.show()
