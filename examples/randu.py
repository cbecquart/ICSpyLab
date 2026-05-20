from icspylab import ICS, plot_ics
from icspylab.distributions import generate_randu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Generate and plot the RANDU dataset

X = generate_randu()
print("X shape:", X.shape)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=2)
plt.savefig("../docs/_static/randu_3d.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Compute and plot the invariant components
ics = ICS(S1="cov", S2="tcovAxis", algorithm="standard")
X_ics = ics.fit_transform(X)
plot_ics(X_ics)

# Compute and plot the principal components
scaler = StandardScaler().set_output(transform="pandas")
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(5, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.title("PCA projection of the simulated dataset \n(first two components)")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.axis("equal")
plt.savefig("../docs/_static/randu_pca.png", dpi=200, bbox_inches="tight")
plt.close()
