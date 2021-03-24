import numpy as np

s = 25
a = 5
v_max = 20
gamma = 0.95
delta = 0.95

xi = 10
N_wedge = 32 * v_max ** 2 * np.log(2 * s * a * 2 ** s / delta) / xi ** 2 / (1 - gamma) ** 2
print(f'N_wedge: {N_wedge}')
