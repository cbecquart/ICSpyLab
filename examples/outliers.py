import numpy as np
from icspylab import ICS, med_crit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import fetch_covtype
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

X, y = fetch_covtype(return_X_y=True, as_frame=True)
s = (y == 2) + (y == 4)
X = X.loc[s]
y = y.loc[s]
y = (y != 2).astype(np.int32)

mask = X.notna().all(axis=1) & y.notna()
X = X.loc[mask]
y = y.loc[mask]

print("X shape:", X.shape)

# Features cleaning
zero_ratio = (X == 0).mean()
cols_to_drop = zero_ratio[zero_ratio > 0.999].index
print("Features to drop (more than 99.9% of 0 values):\n", cols_to_drop)
X = X.drop(cols_to_drop, axis=1)
print("X shape:", X.shape)


# Train test split
X_train, X_other, y_train, y_other = train_test_split(X, y, train_size=0.05, stratify=y, random_state=42)
X_test, _, y_test, _ = train_test_split(X_other, y_other, train_size=0.05, stratify=y_other, random_state=42)

n_samples, n_features = X_train.shape
anomaly_frac = y_train.mean()
print(f"{n_samples} datapoints with {y_train.sum()} anomalies ({anomaly_frac:.02%}) and {n_features} features")


# LOF with and without ICS

lof_ics  = make_pipeline(
    ICS(method_select=med_crit),
    RobustScaler(),
    LocalOutlierFactor(
        n_neighbors=int(n_samples * anomaly_frac),
        novelty=True)
)

lof_ics.fit(X_train)
scores_ics = -lof_ics.decision_function(X_test)

lof_plain = make_pipeline(
    RobustScaler(),
    LocalOutlierFactor(
        n_neighbors=int(n_samples * anomaly_frac),
        novelty=True)
)

lof_plain.fit(X_train)
scores_plain = -lof_plain.decision_function(X_test)

# roc curves
fpr_ics, tpr_ics, _ = roc_curve(y_test, scores_ics)
auc_ics = auc(fpr_ics, tpr_ics)

fpr_plain, tpr_plain, _ = roc_curve(y_test, scores_plain)
auc_plain = auc(fpr_plain, tpr_plain)

plt.figure(figsize=(6, 6))

plt.plot(fpr_ics, tpr_ics, label=f"ICS + LOF (AUC = {auc_ics:.3f})")
plt.plot(fpr_plain, tpr_plain, label=f"LOF only (AUC = {auc_plain:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – LOF with and without ICS")
plt.legend()
plt.grid(True)
plt.show()

# confusion matrices
y_pred_ics = lof_ics.predict(X_test)      # 1=inlier, -1=outlier
y_pred_ics_bin = (y_pred_ics == -1).astype(int)  # 1=outlier, 0=inlier

y_pred_plain = lof_plain.predict(X_test)
y_pred_plain_bin = (y_pred_plain == -1).astype(int)

cm = confusion_matrix(y_test, y_pred_ics_bin)
cm2 = confusion_matrix(y_test, y_pred_plain_bin)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(cm, display_labels=["Inlier", "Outlier"]).plot(ax=axes[0], cmap=plt.cm.Blues)
axes[0].set_title("ICS + LOF")

ConfusionMatrixDisplay(cm2, display_labels=["Inlier", "Outlier"]).plot(ax=axes[1], cmap=plt.cm.Oranges)
axes[1].set_title("LOF only")
plt.show()

# F1 scores
f1_ics = f1_score(y_test, y_pred_ics_bin)
f1_plain = f1_score(y_test, y_pred_plain_bin)

print(f"F1 score ICS + LOF: {f1_ics:.3f}")
print(f"F1 score LOF only: {f1_plain:.3f}")


# IForest with and without ICS

if_ics = make_pipeline(
    ICS(method_select=med_crit),
    IsolationForest(random_state=42)
)

if_ics.fit(X_train)
scores_ics = -if_ics.decision_function(X_test)


if_plain = IsolationForest(random_state=42)

if_plain.fit(X_train)
scores_plain = -if_plain.decision_function(X_test)


fpr_ics, tpr_ics, _ = roc_curve(y_test, scores_ics)
auc_ics = auc(fpr_ics, tpr_ics)

fpr_plain, tpr_plain, _ = roc_curve(y_test, scores_plain)
auc_plain = auc(fpr_plain, tpr_plain)

plt.figure(figsize=(6, 6))

plt.plot(fpr_ics, tpr_ics, label=f"ICS + IF (AUC = {auc_ics:.3f})")
plt.plot(fpr_plain, tpr_plain, label=f"IF only (AUC = {auc_plain:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – IF with and without ICS")
plt.legend()
plt.grid(True)
plt.show()

# confusion_matrix
y_pred_if_ics = if_ics.predict(X_test)      # 1=inlier, -1=outlier
y_pred_if_ics_bin = (y_pred_if_ics == -1).astype(int)  # 1=outlier, 0=inlier

y_pred_if_plain = if_plain.predict(X_test)
y_pred_if_plain_bin = (y_pred_if_plain == -1).astype(int)

cm = confusion_matrix(y_test, y_pred_if_ics_bin)
cm2 = confusion_matrix(y_test, y_pred_if_plain_bin)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(cm, display_labels=["Inlier", "Outlier"]).plot(ax=axes[0], cmap=plt.cm.Blues)
axes[0].set_title("ICS + IF")

ConfusionMatrixDisplay(cm2, display_labels=["Inlier", "Outlier"]).plot(ax=axes[1], cmap=plt.cm.Oranges)
axes[1].set_title("IF only")
plt.show()

f1_if_ics = f1_score(y_test, y_pred_if_ics_bin)
f1_if_plain = f1_score(y_test, y_pred_if_plain_bin)

print(f"F1 score ICS + IF: {f1_if_ics:.3f}")
print(f"F1 score IF only: {f1_if_plain:.3f}")
