import numpy as np

img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

n_row = img.shape[0]
n_col = img.shape[1]

minus = (-1) * np.eye(n_row)
ones = np.eye(n_row)
zeros = np.zeros((1, n_col))
plus = np.append(zeros, ones, axis=0)[:-1, :]

Dx = minus + plus
print(Dx)

Ix = img @ Dx
print(img)
print(Ix)


print("------------------------------------")
minus = (-1) * np.eye(n_col)
ones = np.eye(n_col)
zeros = np.zeros((n_row, 1))
plus = np.append(zeros, ones, axis=1)[:, :-1]

Dy = minus + plus

Iy = Dy @ img
print(Dy)
print(img)
print(Iy)