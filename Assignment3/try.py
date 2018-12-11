import numpy as np
import matplotlib.pyplot as plt


#alphas = [0.95, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999]
alphas = [0.9990, 0.9991, 0.9992, 0.9993, 0.9994, 0.9995, 0.9996, 0.9997, 0.9998, 0.9999]
maxiter = 25000

x = np.linspace(0, maxiter, 1000)
for alpha in alphas:
    plt.plot(x, 90*np.power(alpha, x), label=('%.4f' % alpha))
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Temperature')
plt.show()
