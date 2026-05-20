import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from icspylab import ICS, cov, tcov


p = 4
sigma = 0.2
n = 500


def points_uniform_circle(n, p=2, rayon=np.sqrt(2)):
    x = np.random.normal(size=(n, p))
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return rayon * x / norms


# Generate the data

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

# Plot the original data
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
plt.title("Original dataset (first two dimensions)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis("equal")
plt.savefig("../docs/_static/circle_og.png", dpi=200, bbox_inches="tight")
plt.close()

# PCA
scaler = StandardScaler().set_output(transform="pandas")
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot PCA projection
plt.figure(figsize=(5, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.title("PCA projection of the simulated dataset \n(first two components)")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.axis("equal")
plt.savefig("../docs/_static/circle_pca.png", dpi=200, bbox_inches="tight")
plt.close()

# ICS
ics = ICS(S1="tcov", S2="cov", algorithm="standard")
X_ics = ics.fit_transform(X)

# Plot ICS projection
plt.figure(figsize=(5, 5))
plt.scatter(X_ics[:, 0], X_ics[:, 1], alpha=0.7)
plt.title("ICS projection of the simulated dataset \n(first two components, TCOV–COV scatter pair)")
plt.xlabel("Invariant component 1")
plt.ylabel("Invariant component 2")
plt.axis("equal")
plt.savefig("../docs/_static/circle_ics.png", dpi=200, bbox_inches="tight")
plt.close()
