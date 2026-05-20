from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from icspylab import ICS, plot_ics


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
X_sub = X.sample(frac=0.05, random_state=42)
print("X_sub shape:", X_sub.shape)

plot_ics(
    X_sub,
    col_names=X_sub.columns.tolist(),
    plot_kws={'alpha':0.7}
)

# PCA

scaler = StandardScaler().set_output(transform="pandas")
scaled_X_sub = scaler.fit_transform(X_sub)

pca = PCA()
X_transformed_pca = pca.fit_transform(scaled_X_sub)

plot_ics(
    X_transformed_pca,
    components="first",
    col_names=[f"PC_{i+1}" for i in range(X_transformed_pca.shape[1])],
    plot_kws={'alpha':0.7}
)

kmeans_pca = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X_transformed_pca)
plot_ics(X_transformed_pca, y=kmeans_pca.labels_,
         components="first",
         col_names=[f"PC_{i+1}" for i in range(X_transformed_pca.shape[1])],
         plot_kws={'alpha':0.7})


# ICS

ics = ICS(S1="tcov", S2="cov", center=True)
X_transformed_ics = ics.fit_transform(X_sub)
plot_ics(X_transformed_ics, components="first", plot_kws={'alpha':0.7})

kmeans_ics = KMeans(n_clusters=7, random_state=0, n_init="auto").fit(X_transformed_ics)
plot_ics(X_transformed_ics,
         components="first",
         y=kmeans_ics.labels_,
         plot_kws={'alpha':0.7})
