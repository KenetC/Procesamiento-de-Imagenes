import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from generar_matriz import matriz_txt
from sklearn.decomposition import PCA

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

# ----- Testing autovectores ortonormales -----
# print(np.linalg.norm(U[:, 0]))

V = np.dot(A[0], U)

# ----- Autovalores -----
autovalores = np.loadtxt('datos_2dpca/autovalores.txt')
plt.plot(autovalores)
plt.show()

# ----- Testing Eigenfaces -----
# Yj * Xj.T = subimagen (eigenface?)

#f, axs = plt.subplots(2, 5, figsize=(12, 12))
#for i, ax in enumerate(axs.flatten()):
#    subimagen = np.outer(V[:, i], U.T[i])
#    ax.imshow(subimagen, cmap=plt.cm.gray)
#    ax.axis('off')
#plt.show()

# ----- Reconstrucción -----
# A = V * U.T

#f, axs = plt.subplots(3, 3, figsize=(12, 12))
#for i, ax in enumerate(axs.flatten()):
#    k = (i + 1) * 10
#    Vk = V[:, :k]
#    Uk = U[:, :k]
#    A_new = np.dot(Vk, Uk.T)
#    ax.imshow(A_new, cmap=plt.cm.gray)
#    ax.axis('off')
#    ax.set_title(f'k = {k}')
#plt.show()

# ----- Testing Reconstrucción -----
# Vk = V[:, :10]
# Uk = U[:, :10]
# A_new = np.dot(Vk, Uk.T)
# plt.imshow(A_new, cmap=plt.cm.gray)
# plt.show()

# ----- Matriz de similaridad -----

k = 3 
Z1_2dpca = np.zeros((410,112*k))

for i in range(0, 410):
	Z1_2dpca[i] = calcular_z_2dpca(k,i).flatten()


k = 90 
Z2_2dpca = np.zeros((410,112*k))


for i in range(0, 410):
	Z2_2dpca[i] = calcular_z_2dpca(k,i).flatten()


plt.pcolor(np.corrcoef(Z1_2dpca), cmap='GnBu')
plt.colorbar()
plt.show()

plt.pcolor(np.corrcoef(Z2_2dpca), cmap='GnBu')
plt.colorbar()
plt.show()

z1_2dpca = np.corrcoef(Z1_2dpca)
z2_2dpca = np.corrcoef(Z2_2dpca)

# ----- Metrica mismo -----

def met_mism (z):
    sum = 0
    i_aux = 0
    j_aux = 410
    for i in range (len(z)):
        for j in range(10, 0, -1):
            sum += z[i][j_aux -j]
        i_aux = i_aux + 1
        if (i_aux == 10):
            j_aux = j_aux - 10
            i_aux =0
    t = sum/4100
    return t

print(met_mism(z1_2dpca))
print(met_mism(z2_2dpca))

# ----- Metrica distinto -----

def met_dist (z):
    sum = 0
    i_aux = 0
    j_aux = 400
    for i in range (len(z)):
        for j in range (len(z)):
            if (j<j_aux or j>j_aux+9):
                sum += z[i][j]
        i_aux = i_aux + 1
        if (i_aux == 10):
            j_aux = j_aux - 10
            i_aux =0
    t = sum/(410*410-4100)
    return t

print(met_dist(z1_2dpca))
print(met_dist(z2_2dpca))

# ------ Medimos la calidad de la similaridad entre imágenes ------

ks = np.ceil(np.logspace(1,np.log10(92),20)).astype(int)

mismo_2dpca = []
dist_2dpca = []

for k in ks:
    if k<92:
        Z = np.zeros((410,112*k))
    else:
        Z = np.zeros((410,112*92))
    for i in range(0, 410):
        Z[i] = calcular_z_2dpca(k,i).flatten()
    mismo_2dpca.append(met_mism(np.corrcoef(Z)))
    dist_2dpca.append(met_dist(np.corrcoef(Z)))
print(mismo_2dpca)
print(dist_2dpca)

plt.figure()
plt.plot(ks,mismo_2dpca, color='blue', label='Métrica mismos')
plt.plot(ks,dist_2dpca, color='red', label= 'Métrica distintos')
plt.xlabel('valor de k')
plt.ylabel('calidad de la similaridad')
plt.title('2DPCA')
plt.legend()
plt.show()