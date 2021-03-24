import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)
alpha = np.ones(25) * 0.1

fig, axs = plt.subplots(4, 4, sharey=True)
fig.suptitle('Samples from the dirichlet distribution with 25 objects and alpha=0.1', fontsize=16)
for i in range(4):
    for j in range(4):
        probabilities = np.random.dirichlet((alpha))
        axs[i, j].plot(probabilities, 'bo')
