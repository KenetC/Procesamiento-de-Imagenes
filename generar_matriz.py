import numpy as np
import os

D = np.diag([5.0, 4.0, 3.0, 2.0, 1.0])

v = np.ones((D.shape[0], 1))

v = v / np.linalg.norm(v)
# print(v)

# Matriz de Householder
B = np.eye(D.shape[0]) - 2 * (v @ v.T)
# print(B)

# Matriz a diagonalizar
M = B.T @ D @ B
# print(M.shape)
# print(M)

def matriz_txt(A, niter, eps):
    direccion_actual = os.getcwd()
    path = os.path.join(direccion_actual, 'input_data.txt')

    if os.path.isfile(path):
        os.remove(path)
    with open('input_data.txt','a') as f:
        f.write(f"{A.shape[0]} {A.shape[1]} {niter} {eps}\n")
        np.savetxt(f,A, newline="\n")

matriz_txt(M, 10000, 0.00000001)