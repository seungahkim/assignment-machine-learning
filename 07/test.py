import numpy as np

a = np.array(range(3)).reshape((3, 1))
b = np.ones((3, 10))

print(a * b)
