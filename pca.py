import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import decomposition

# GENERATE DATA
iris = datasets.load_iris()

X = iris.data
y = iris.target

# CENTER DATA
X_centered = np.ndarray(shape=(150, 4), dtype=float)
means = np.mean(X, axis=0)
for i in range(150):
    X_centered[i] = X[i] - means
X_centered = X_centered.T
# PROJECT DATA
cov_mat = np.cov(X_centered)
eig_values, eig_vectors = np.linalg.eig(cov_mat)
indices = sorted(range(len(eig_values)), key=eig_values.__getitem__, reverse=True)
index_1 = indices[0]
index_2 = indices[1]
print(f"this is our 2D subspace:\n {eig_vectors[:, [index_1,index_2]]}")
w_mat = eig_vectors[:, [index_1, index_2]]
projected_data = np.dot(X_centered.T, w_mat)

# visualize projected data
plt.plot(projected_data[y == 0, 0], projected_data[y == 0, 1], 'bo', label='Setosa')
plt.plot(projected_data[y == 1, 0], projected_data[y == 1, 1], 'go', label='Versicolour')
plt.plot(projected_data[y == 2, 0], projected_data[y == 2, 1], 'ro', label='Virginica')
plt.legend(loc=0)
plt.show()

# RESTORE DATA
restored_data = np.dot(w_mat, projected_data.T)
for i in range(150):
    restored_data.T[i] = restored_data.T[i] + means

pca = decomposition.PCA(n_components=2)
# class method "fit" for our centered data
pca.fit(X_centered.T)
# make a projection
X_pca = pca.transform(X_centered.T)

plt.plot(X_pca[y == 0, 0], X_pca[y == 0, 1], 'bo', label='Setosa')
plt.plot(X_pca[y == 1, 0], X_pca[y == 1, 1], 'go', label='Versicolour')
plt.plot(X_pca[y == 2, 0], X_pca[y == 2, 1], 'ro', label='Virginica')
plt.legend(loc=0)
plt.show()
