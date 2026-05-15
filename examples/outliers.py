import numpy as np
from icspylab import ICS, median_crit, plot_ics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import fetch_covtype
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

X, y = fetch_covtype(return_X_y=True, as_frame=True)
s = (y == 2) + (y == 4)
X = X.loc[s]
y = y.loc[s]
y = (y != 2).astype(np.int32)
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


# LOF

def fit_predict_scores(model, X_train, X_test):
    model.fit(X_train)
    scores = -model.decision_function(X_test)
    # Predictions
    y_pred = model.predict(X_test)  # 1=inlier, -1=outlier
    y_pred_bin = (y_pred == -1).astype(int)  # 1=outlier, 0=inlier
    return scores, y_pred_bin

lof_plain = Pipeline([
    ("scaler", RobustScaler()),
    ("lof", LocalOutlierFactor(n_neighbors=int(n_samples * anomaly_frac), novelty=True))
])

scores_lof_plain, y_pred_plain_bin = fit_predict_scores(lof_plain, X_train, X_test)


# LOF with ICS

lof_ics = Pipeline([
    ("ics", ICS(method_select=median_crit)),
    ("scaler", RobustScaler()),
    ("lof", LocalOutlierFactor(n_neighbors=int(n_samples * anomaly_frac), novelty=True))
])

scores_lof_ics, y_pred_ics_bin = fit_predict_scores(lof_ics, X_train, X_test)

# Optional: hyperparameter tuning via cross-validation
# Note: this is not used in this example due to the extreme class imbalance
#
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#
# def get_f1_score(estimator, X, y):
#     y_pred = estimator.predict(X)
#     y_pred_bin = (y_pred == -1).astype(int)
#     return f1_score(y, y_pred_bin)
#
# param_grid = {
#     "ics__select_args": [{"nb_select": 5}, {"nb_select": 10}, {"nb_select": 20}, {"nb_select": n_features - 1}]
# }
#
# lof_ics_grid = GridSearchCV(
#     lof_ics,
#     param_grid,
#     scoring= get_f1_score,
#     cv=cv
# )
#
# lof_ics_grid.fit(X_train, y_train)
# print(lof_ics_grid.best_params_)
#
# scores_lof_ics, y_pred_ics_bin = fit_predict_scores(lof_ics_grid.best_estimator_, X_train, X_test)


# IForest

if_plain = Pipeline([
    ("if", IsolationForest(random_state=42))
])

scores_if_plain, y_pred_if_plain_bin = fit_predict_scores(if_plain, X_train, X_test)


# IForest with ICS

if_ics = Pipeline([
    ("ics", ICS(method_select=median_crit)),
    ("if", IsolationForest(random_state=42))
])

scores_if_ics, y_pred_if_ics_bin = fit_predict_scores(if_ics, X_train, X_test)


# ROC curves

def plot_roc(ax, curves, title, y_test):
    for label, scores in curves.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Subplot 1 : LOF
plot_roc(ax=axes[0],
         curves={"ICS + LOF": scores_lof_ics, "LOF only": scores_lof_plain},
         title="ROC Curve – LOF",
         y_test=y_test)

# Subplot 2 : Isolation Forest
plot_roc(ax=axes[1],
         curves={"ICS + IF": scores_if_ics,  "IF only": scores_if_plain},
         title="ROC Curve – Isolation Forest",
         y_test=y_test)

plt.tight_layout()
plt.savefig("../docs/_static/outliers_ROC.png", dpi=200, bbox_inches="tight")
plt.close()


# Confusion matrices

def plot_confusion(ax, y_pred_bin, title, cmap):
    cm = confusion_matrix(y_test, y_pred_bin)
    ConfusionMatrixDisplay(cm, display_labels=["Inlier", "Outlier"]).plot(
        ax=ax, cmap=cmap, colorbar=False
    )
    ax.set_title(title)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plot_confusion(ax=axes[0, 0], y_pred_bin=y_pred_ics_bin, title="ICS + LOF", cmap=plt.cm.Blues)
plot_confusion(ax=axes[1, 0], y_pred_bin=y_pred_plain_bin, title="LOF only", cmap=plt.cm.Oranges)
plot_confusion(ax=axes[0, 1], y_pred_bin=y_pred_if_ics_bin, title="ICS + IF", cmap=plt.cm.Blues)
plot_confusion(ax=axes[1, 1], y_pred_bin=y_pred_if_plain_bin, title="IF only", cmap=plt.cm.Oranges)

plt.tight_layout()
plt.savefig("../docs/_static/outliers_CM.png", dpi=200, bbox_inches="tight")
plt.close()


# F1 scores

f1_plain = f1_score(y_test, y_pred_plain_bin)
f1_ics = f1_score(y_test, y_pred_ics_bin)
f1_if_plain = f1_score(y_test, y_pred_if_plain_bin)
f1_if_ics = f1_score(y_test, y_pred_if_ics_bin)

print(f"F1 score LOF only: {f1_plain:.3f}")
print(f"F1 score ICS + LOF: {f1_ics:.3f}")
print(f"F1 score IF only: {f1_if_plain:.3f}")
print(f"F1 score ICS + IF: {f1_if_ics:.3f}")
