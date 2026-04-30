from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from icspylab import ICS, cov, tcov, tM, cov4, tcovAxis, mcd, plot_ics
import numpy as np
import pandas as pd


X, y = fetch_covtype(return_X_y=True, as_frame=True)
scaler = StandardScaler().set_output(transform="pandas")
s = (y == 2)
X = X.loc[s]
y = y.loc[s]

mask = X.notna().all(axis=1) & y.notna()
X = X.loc[mask]
y = y.loc[mask]

print("X shape:", X.shape)

# Features cleaning
zero_ratio = (X == 0).mean()
cols_to_drop = zero_ratio[zero_ratio > 0.95].index
print("Features to drop (more than 95% of 0 values):\n", cols_to_drop)
X = X.drop(cols_to_drop, axis=1)
X = X.drop("Hillshade_3pm", axis=1)
print("X shape:", X.shape)


# Train test split
X_train, X_other, y_train, y_other = train_test_split(X, y, train_size=0.03, stratify=y, random_state=42)
X_test, _, y_test, _ = train_test_split(X_other, y_other, train_size=0.01, stratify=y_other, random_state=42)
print("X_train shape:", X_train.shape)

scaled_X_train = scaler.fit_transform(X_train)



pca = PCA(n_components=3).fit(X_train)
scaled_pca = PCA(n_components=3).fit(scaled_X_train)
X_train_transformed_pca = pca.transform(X_train)
X_train_std_transformed_pca = scaled_pca.transform(scaled_X_train)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

ax1.scatter(
    x=X_train_transformed_pca[:, 0],
    y=X_train_transformed_pca[:, 1],
    alpha=0.7
)

ax2.scatter(
    x=X_train_std_transformed_pca[:, 0],
    y=X_train_std_transformed_pca[:, 1],
    alpha=0.7
)

ax1.set_title("Unscaled training dataset after PCA")
ax2.set_title("Standardized training dataset after PCA")

for ax in (ax1, ax2):
    ax.set_xlabel("1st principal component")
    ax.set_ylabel("2nd principal component")
    ax.legend(loc="upper right")
    ax.grid()

fig.show()

plot_ics(X_train_std_transformed_pca, plot_kws={'alpha':0.7})



ics = ICS(S1=tcov, S2=cov).fit(X_train)
# ics = ICS(S1=cov, S2=cov4, algorithm="QR").fit(X_train)
scaled_ics = ICS(S1=tcov, S2=cov).fit(scaled_X_train)
X_train_transformed_ics = ics.transform(X_train)
#X_train_std_transformed = scaled_ics.transform(scaled_X_train)

plot_ics(X_train_transformed_ics, plot_kws={'alpha':0.7})



from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=7, random_state=0, n_init="auto").fit(X_train_transformed_ics[:, [1, 2]])
centroids_ics = kmeans.cluster_centers_

centroids_orig = centroids_ics @ ics.components_[[1, 2], :]
plot_ics(centroids_orig[:, 9:], plot_kws={'alpha':0.7})

# X_train['cluster'] = kmeans.labels_
# X_train.groupby('cluster').mean()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=ics.feature_names_in_)
X_train_scaled['cluster'] = kmeans.labels_

X_train_scaled.groupby('cluster').mean()

print("done")