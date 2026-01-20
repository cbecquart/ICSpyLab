import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from icspylab import ICS, cov, covW, cov4, mcd, tcov, tcov2
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Aplatir les images 28x28 → 784
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Normalisation (fortement recommandée)
X_train_flat = X_train_flat / 255.0
X_test_flat = X_test_flat / 255.0

pca = PCA(n_components=0.95, random_state=42)  # 95% variance
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)

print("Nombre de composantes retenues :", pca.n_components_)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA on MNIST")
plt.show()

pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_train_flat)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, s=1, cmap="tab10")
plt.colorbar()
plt.title("MNIST projected with PCA")
plt.show()

ics = ICS(S1=cov, S2=tcov2, algorithm='standard')
ics.fit_transform(X_train_flat)
ics.plot()