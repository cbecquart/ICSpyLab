from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from icspylab import ICS, cov, tcov, plot_ics
import numpy as np
import pandas as pd


# Import data
X, y = fetch_covtype(return_X_y=True, as_frame=True)
print(y.value_counts())
s = (y == 2)
X = X.loc[s]
y = y.loc[s]

print("X shape:", X.shape)

# Features cleaning
zero_ratio = (X == 0).mean()
cols_to_drop = zero_ratio[zero_ratio > 0.95].index
print("Features to drop (more than 95% of 0 values):\n", cols_to_drop)
X = X.drop(cols_to_drop, axis=1)
print("X shape:", X.shape)

# Subsample the data
X_sub, _, y_train, _ = train_test_split(X, y, train_size=0.05, stratify=y, random_state=42)
print("X_sub shape:", X_sub.shape)


# PCA

scaler = StandardScaler().set_output(transform="pandas")
scaled_X_sub = scaler.fit_transform(X_sub)

pca = PCA(n_components=3)
X_transformed_pca = pca.fit_transform(scaled_X_sub)

plot_ics(
    X_transformed_pca,
    col_names=[f"PC_{i+1}" for i in range(X_transformed_pca.shape[1])],
    plot_kws={'alpha':0.7}
)

# ICS

ics = ICS(S1=tcov, S2=cov)
X_transformed_ics = ics.fit_transform(X_sub)
plot_ics(X_transformed_ics, plot_kws={'alpha':0.7})


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=7, random_state=0, n_init="auto").fit(X_transformed_ics[:, [1, 2]])
centroids_ics = kmeans.cluster_centers_

centroids_orig = centroids_ics @ ics.components_[[1, 2], :]
plot_ics(centroids_orig[:, 9:], plot_kws={'alpha':0.7})

# X_sub['cluster'] = kmeans.labels_
# X_sub.groupby('cluster').mean()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_sub)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=ics.feature_names_in_)
X_train_scaled['cluster'] = kmeans.labels_

X_train_scaled.groupby('cluster').mean()

print("done")