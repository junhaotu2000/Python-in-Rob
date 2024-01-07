import numpy as np


x = np.random.rand(10000, 10000)
y = np.random.rand(10000, 10000)
z = x @ y

print(z)
