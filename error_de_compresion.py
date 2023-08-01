import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from generar_matriz import matriz_txt

imgs = []
for path in sorted(list(Path('ImagenesCaras').rglob('*/*.pgm'))):
    imagen_fila = plt.imread(path)[::2, ::2]/255
    imgs.append(imagen_fila.flatten())
X = np.stack(imgs)
m, n = X.shape

Xc = X.copy()
for i in range(len(X)):
    mu = np.mean(X[i])
    Xc[i] = X[i] - mu

C = np.dot(Xc.T, Xc) / (n - 1)

matriz_txt(C, 10000, 0.000000001)

V = np.loadtxt('datos_pca/autovectores.txt')

def calcular_z_pca(k):
    Vk = V[:, :k]
    Z = X.dot(Vk)
    return Z, Vk

k = 400
Z, Vk = calcular_z_pca(k)
X_pca = Z.dot(Vk.T)

#mean absolute error (MAE)
def calc_mae(X_orig, X_reconst):
    return np.mean(np.abs(X_orig - X_reconst))


mae_pca = []
for i in range(len(X)):
    i_pca = X_pca[i].reshape(56,46)
    i_or = X[i].reshape(56,46)
    m = calc_mae(i_or, i_pca)
    mae_pca.append(m)



plt.plot(mae_pca)
plt.xlabel('Número de Imágen')
plt.ylabel('EAM')
plt.title('Error Absoluto Medio PCA')
plt.show()


paths = []
A = []
A_promedio = plt.imread('ImagenesCaras/s1/1.pgm')[::, ::]/255
for path in sorted(list(Path('ImagenesCaras').rglob('*/*.pgm'))):
    paths.append(path)
    imagen = plt.imread(path)[::, ::]/255
    A.append(imagen)
    A_promedio = A_promedio + imagen
A_promedio = A_promedio / len(A)
a, b = A_promedio.shape

G = np.zeros((b, b))
for i in range(len(A)):
    Aj = A[i]
    G = G + np.dot((Aj - A_promedio).T, (Aj - A_promedio))
G = G / len(A)

matriz_txt(G, 10000, 0.000000000001)

U = np.loadtxt('datos_2dpca/autovectores.txt')

def calcular_z_2dpca(k, i):
    Uk = U[:, :k]
    Ai = A[i]
    Zi = np.dot(Ai, Uk)
    return Zi

k = 92

mae_2dpca = []
for i in range(len(A)):
    A_2dpca = calcular_z_2dpca(k, i)  
    m = calc_mae(A[i], A_2dpca)
    mae_2dpca.append(m)

plt.plot(mae_2dpca)
plt.xlabel('Número de Imágen')
plt.ylabel('EAM')
plt.title('Error Absoluto Medio 2DPCA')
plt.show()


# Version Boxplot
fig, ax = plt.subplots()
ax.boxplot(mae_pca, positions=[1], widths=0.6)
ax.boxplot(mae_2dpca, positions=[2], widths=0.6)
ax.set_xticklabels(['PCA', '2DPCA'])
ax.set_ylabel('EAM')
plt.title('Comparación Error Absoluto Medio')
plt.show()