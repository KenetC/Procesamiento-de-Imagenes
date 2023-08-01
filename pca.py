import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from generar_matriz import matriz_txt
from sklearn.decomposition import PCA

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

# ----- Autovalores -----

autovalores = np.loadtxt('datos_pca/autovalores.txt')
plt.plot(autovalores)
plt.show()

# ----- Eigenfaces -----
# Agarrar un autovector y redimensionarlo al tamaño de la imagen original

#V = np.loadtxt('datos_pca/autovectores.txt')
#f, axs = plt.subplots(2, 5, figsize=(12, 12))
#for i, ax in enumerate(axs.flatten()):
#    ax.imshow(V[:, i].reshape(56, 46), cmap=plt.cm.gray)
#    ax.axis('off')
#plt.show()

# ----- Reconstrucción -----
# Z = X * Vk
# X_new = Z * Vk
""""
f, axs = plt.subplots(5, 5, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
    k = (i + 1) * 20
    Z, Vk = calcular_z_pca(k)
    X_new = Z.dot(Vk.T)
    ax.imshow(X_new[0].reshape(56, 46), cmap=plt.cm.gray)
    ax.axis('off')
    ax.set_title(f'k = {k}', size=8)
plt.show()
"""
# ----- Testing Reconstrucción -----
# Vk = V[:, :2000]
# Z = X.dot(Vk)

# X_new = Z.dot(Vk.T)

# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
# ax1.imshow(X_new[0].reshape(56, 46), cmap=plt.cm.gray)
# ax2.imshow(X[0].reshape(56, 46), cmap=plt.cm.gray)
# plt.tight_layout()
# plt.show()

# ----- Testing -----
# w2, V2 = np.linalg.eig(C)
# w3, V3 = np.linalg.eigh(C)

# print(V[0])
# print(V2[0])
# print(V3[0])

# pca = PCA(410).fit(X)
# print(pca.components_.shape)

# print(pca.components_[0].shape)
# print(pca.components_[0])
# print(V[:, 0].shape)
# print(V[:, 0])

# f, axs = plt.subplots(2, 5, figsize=(12, 12))
# for i, ax in enumerate(axs.flatten()):
#     ax.imshow(pca.components_[i].reshape(56, 46), cmap=plt.cm.gray)
#     ax.axis('off')
# plt.tight_layout()
# plt.show()


# ----- Matriz de similaridad -----

plt.pcolor(np.corrcoef(Xc), cmap='GnBu')
plt.colorbar()
plt.show()

Z1_pca = calcular_z_pca(10)[0]
Z2_pca = calcular_z_pca(400)[0]


plt.pcolor(np.corrcoef(Z1_pca), cmap='GnBu')
plt.colorbar()
plt.show()

plt.pcolor(np.corrcoef(Z2_pca), cmap='GnBu')
plt.colorbar()
plt.show()

z1_pca = np.corrcoef(Z1_pca)
z2_pca = np.corrcoef(Z2_pca)

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

# ----- Metrica distinto -----

def met_dist (z):
    sum = 0
    i_aux = 0
    j_aux = 400
    cant_im = 0
    for i in range (len(z)):
        for j in range (len(z)):
            if (j<j_aux or j>j_aux+9):
                sum += z[i][j]
                cant_im += 1
        i_aux = i_aux + 1
        if (i_aux == 10):
            j_aux = j_aux - 10
            i_aux =0
    t = sum/cant_im
    return t

# ------ Medimos la calidad de la similaridad entre imágenes ------

ks = np.ceil(np.logspace(1,np.log10(600),30)).astype(int)
print(ks)
print(met_mism(np.corrcoef(Xc)))
print(met_dist(np.corrcoef(Xc)))
mismo_pca = []
dist_pca = []

for k in ks:
    C = np.corrcoef(calcular_z_pca(k)[0])
    mismo_pca.append(met_mism(C))
    dist_pca.append(met_dist(C))
print(mismo_pca)
print(dist_pca)

plt.figure()
plt.plot(ks,mismo_pca, color='blue', label='Métrica mismos')
plt.plot(ks,dist_pca, color='red', label= 'Métrica distintos')
plt.xlabel('valor de k')
plt.ylabel('calidad de la similaridad')
plt.title('PCA')
plt.legend()
plt.show()