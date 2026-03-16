import numpy as np
import matplotlib.pyplot as plt
from icspylab import ICS, cov, covW, cov4, mcd, tcov, tcovAxis, plot_ics
from sklearn.decomposition import PCA


def generate_randu_dataset(n_points=400, seed=1.0):

    def randu():
        nonlocal seed
        seed = ((2 ** 16 + 3) * seed) % (2 ** 31)
        return seed / (2 ** 31)

    x = np.empty((n_points, 3), dtype=float)

    for i in range(n_points):
        U = np.array([randu() for _ in range(5)])
        x[i, :] = np.round(U[:3], 6)

    return x


def generate_randu(n_points=1000, seed=1):
    m = 2**31
    a = 65539

    x = np.empty(n_points * 3, dtype=np.int64)
    x[0] = seed

    for i in range(1, len(x)):
        x[i] = (a * x[i - 1]) % m

    # Regroupement en triplets
    data = x.reshape(-1, 3).astype(float)

    # Normalisation dans [0, 1]
    data /= m

    return data

# X = generate_randu(n_points=400)


# Generate and plot the RANDU dataset

X = generate_randu_dataset()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=2)
plt.savefig("../docs/_static/randu_3d.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Compute and plot the invariant components
# ics = ICS(S1=cov, S2=covW, algorithm='standard', S2_args={'alpha': 1, 'cf': 2})
ics = ICS(S1=cov, S2=tcovAxis, algorithm='standard')
X_ics = ics.fit_transform(X)
plot_ics(X_ics)
ics.plot_kurtosis()

# Compute and plot the principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(5, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.title("PCA projection of the simulated dataset \n(first two components)")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.axis("equal")
plt.savefig("../docs/_static/randu_pca.png", dpi=200, bbox_inches="tight")
plt.close()

ics.describe()
