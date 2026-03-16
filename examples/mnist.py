from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tests.settings import algorithm

X, y = load_wine(return_X_y=True, as_frame=True)
scaler = StandardScaler().set_output(transform="pandas")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
scaled_X_train = scaler.fit_transform(X_train)


import matplotlib.pyplot as plt
from icspylab import ICS, cov, tcov, tM, cov4, tcovAxis

# from sklearn.neighbors import KNeighborsClassifier
#
# X_plot = X[["proline", "hue"]]
# X_plot_scaled = scaler.fit_transform(X_plot)
# clf = KNeighborsClassifier(n_neighbors=20)




import pandas as pd

from sklearn.decomposition import PCA

pca = PCA(n_components=2, svd_solver="covariance_eigh").fit(X_train)
scaled_pca = PCA(n_components=2, svd_solver="covariance_eigh").fit(scaled_X_train)
X_train_transformed = pca.transform(X_train)
X_train_std_transformed = scaled_pca.transform(scaled_X_train)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

target_classes = range(0, 3)
colors = ("blue", "red", "green")
markers = ("^", "s", "o")

for target_class, color, marker in zip(target_classes, colors, markers):
    ax1.scatter(
        x=X_train_transformed[y_train == target_class, 0],
        y=X_train_transformed[y_train == target_class, 1],
        color=color,
        label=f"class {target_class}",
        alpha=0.5,
        marker=marker,
    )

    ax2.scatter(
        x=X_train_std_transformed[y_train == target_class, 0],
        y=X_train_std_transformed[y_train == target_class, 1],
        color=color,
        label=f"class {target_class}",
        alpha=0.5,
        marker=marker,
    )

ax1.set_title("Unscaled training dataset after PCA")
ax2.set_title("Standardized training dataset after PCA")

for ax in (ax1, ax2):
    ax.set_xlabel("1st principal component")
    ax.set_ylabel("2nd principal component")
    ax.legend(loc="upper right")
    ax.grid()

# _ = plt.tight_layout()
fig.show()

###
from icspylab import mcd
ics = ICS(S1=tcov, S2=cov, algorithm="standard").fit(X_train)
scaled_ics = ICS(S1=tcov, S2=cov, algorithm="eigh").fit(X_train)

X_train_transformed = ics.transform(X_train)
X_train_std_transformed = scaled_ics.transform(X_train)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

target_classes = range(0, 3)
colors = ("blue", "red", "green")
markers = ("^", "s", "o")

for target_class, color, marker in zip(target_classes, colors, markers):
    ax1.scatter(
        x=X_train_transformed[y_train == target_class, 0],
        y=X_train_transformed[y_train == target_class, 1],
        color=color,
        label=f"class {target_class}",
        alpha=0.5,
        marker=marker,
    )

    ax2.scatter(
        x=X_train_std_transformed[y_train == target_class, 0],
        y=X_train_std_transformed[y_train == target_class, 1],
        color=color,
        label=f"class {target_class}",
        alpha=0.5,
        marker=marker,
    )

ax1.set_title("Unscaled training dataset after ICS")
ax2.set_title("Standardized training dataset after ICS")

for ax in (ax1, ax2):
    ax.set_xlabel("1st IC")
    ax.set_ylabel("2nd IC")
    ax.legend(loc="upper right")
    ax.grid()

#_ = plt.tight_layout()
fig.show()

###

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(solver='eigen', n_components=2)
X_lda = lda.fit_transform(X_train, y_train)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

target_classes = range(0, 3)
colors = ("blue", "red", "green")
markers = ("^", "s", "o")

for target_class, color, marker in zip(target_classes, colors, markers):
    ax1.scatter(
        x=X_lda[y_train == target_class, 0],
        y=X_lda[y_train == target_class, 1],
        color=color,
        label=f"class {target_class}",
        alpha=0.5,
        marker=marker,
    )

    ax2.scatter(
        x=X_lda[y_train == target_class, 0],
        y=X_lda[y_train == target_class, 1],
        color=color,
        label=f"class {target_class}",
        alpha=0.5,
        marker=marker,
    )

ax1.set_title("Unscaled training dataset after ICS")
ax2.set_title("Standardized training dataset after ICS")

for ax in (ax1, ax2):
    ax.set_xlabel("1st IC")
    ax.set_ylabel("2nd IC")
    ax.legend(loc="upper right")
    ax.grid()

#_ = plt.tight_layout()
fig.show()





import numpy as np

from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline

Cs = np.logspace(-5, 5, 20)

unscaled_clf = make_pipeline(
    pca, LogisticRegressionCV(Cs=Cs)
)
unscaled_clf.fit(X_train, y_train)

scaled_clf = make_pipeline(
    scaler,
    pca,
    LogisticRegressionCV(Cs=Cs),
)
scaled_clf.fit(X_train, y_train)

print(f"Optimal C for the unscaled PCA: {unscaled_clf[-1].C_:.4f}\n")
print(f"Optimal C for the standardized data with PCA: {scaled_clf[-1].C_:.2f}")



from sklearn.metrics import accuracy_score, log_loss

y_pred = unscaled_clf.predict(X_test)
y_pred_scaled = scaled_clf.predict(X_test)
y_proba = unscaled_clf.predict_proba(X_test)
y_proba_scaled = scaled_clf.predict_proba(X_test)

print("Test accuracy for the unscaled PCA")
print(f"{accuracy_score(y_test, y_pred):.2%}\n")
print("Test accuracy for the standardized data with PCA")
print(f"{accuracy_score(y_test, y_pred_scaled):.2%}\n")
print("Log-loss for the unscaled PCA")
print(f"{log_loss(y_test, y_proba):.3}\n")
print("Log-loss for the standardized data with PCA")
print(f"{log_loss(y_test, y_proba_scaled):.3}")


S1_X, S1_X_inv_sqrt = self._compute_first_scatter(X)
S2_X = self._compute_second_scatter(X)
S2_Y = multi_dot([S1_X_inv_sqrt, S2_X.scatter, S1_X_inv_sqrt])
S2_Y_eigenval, S2_Y_eigenvect = np.linalg.eigh(S2_Y.scatter)
S2_Y_eigenval, S2_Y_eigenvect = sort_eigenvalues_eigenvectors(S2_Y_eigenval, S2_Y_eigenvect)
gen_kurtosis = S2_Y_eigenval
W = np.dot(S2_Y_eigenvect.T, S1_X_inv_sqrt)