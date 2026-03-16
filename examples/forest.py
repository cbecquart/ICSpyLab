from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


X, y = fetch_covtype(return_X_y=True, as_frame=True)
scaler = StandardScaler().set_output(transform="pandas")
s = (y == 2) + (y == 3) + (y == 4)
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
X_train, X_other, y_train, y_other = train_test_split(X, y, train_size=0.05, stratify=y, random_state=42)
X_test, _, y_test, _ = train_test_split(X_other, y_other, train_size=0.05, stratify=y_other, random_state=42)
print("X_train shape:", X_train.shape)

scaled_X_train = scaler.fit_transform(X_train)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit(X_train)
scaled_pca = PCA(n_components=2).fit(scaled_X_train)
X_train_transformed = pca.transform(X_train)
X_train_std_transformed = scaled_pca.transform(scaled_X_train)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

target_classes = (7, 4, 5)
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


from icspylab import ICS, cov, tcov, tM, cov4, tcovAxis, mcd
ics = ICS(S1=cov, S2=cov4, algorithm="QR").fit(X_train)
#scaled_ics = ICS(S1=tcov, S2=cov, algorithm="whiten").fit(scaled_X_train)
X_train_transformed = ics.transform(X_train)
#X_train_std_transformed = scaled_ics.transform(scaled_X_train)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

target_classes = (2, 4, 5)
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
        x=X_train_transformed[y_train == target_class, -1],
        y=X_train_transformed[y_train == target_class, -2],
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


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(solver='eigen', n_components=2)
X_lda = lda.fit_transform(X_train, y_train)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

target_classes = (2, 3, 4)
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


print("done")