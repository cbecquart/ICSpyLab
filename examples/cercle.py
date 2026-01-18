import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from icspylab import ICS, cov, tcov

def points_uniform_circle(n, p=2, rayon=np.sqrt(2)):
    x = np.random.normal(size=(n, p))
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return rayon * x / norms


p = 4
sigma = 0.2
n = 500

rng = np.random.default_rng(seed=0)

signal = points_uniform_circle(n, p=2, rayon=np.sqrt(2))

X1 = signal + rng.multivariate_normal(
    mean=np.zeros(2),
    cov=sigma**2 * np.eye(2),
    size=n
)

X2 = rng.multivariate_normal(
    mean=np.zeros(p-2),
    cov=(1 + sigma**2) * np.eye(p-2),
    size=n
)

X = np.hstack((X1, X2))

plt.scatter(X[:, 0], X[:, 1])
plt.show()

# PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.show()

# ICS

ics = ICS(S1=tcov, S2=cov, algorithm='standard')
ics.fit_transform(X)
plt.scatter(ics.scores_[:, 0], ics.scores_[:, 1])
plt.show()
