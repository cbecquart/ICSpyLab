# Outlier detection

This example illustrates how ICS can be a relevant preprocessing step for an 
outlier detection problem. It also shows how easy it is to integrate ICS in a
scikit learn pipeline. 

The outlier detection algorithms we will consider are [Local Outlier Factor](https://scikit-learn.org/stable/modules/outlier_detection.html#local-outlier-factor) 
(LOF) and [Isolation Forest](https://scikit-learn.org/stable/modules/outlier_detection.html#isolation-forest)
(IForest), both implemented in scikit learn. 

The problem we consider comes from the Scikit learn's [Evaluation of outlier detection estimators](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_outlier_detection_bench.html#forest-covertypes-dataset) 
example.
The dataset is the [Forest covertypes](https://scikit-learn.org/stable/datasets/real_world.html#covtype-dataset) 
dataset. It contains patches of forest and the target is the dominant species of tree in the patch.
There are 54 features whose description is available online. 
The prediction of the target variable is a multiclass classification problem, as there are 7
covertypes. 
However, following Scikit learn's [Evaluation of outlier detection estimators](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_outlier_detection_bench.html#forest-covertypes-dataset) 
example, we will adapt the dataset to make it an outlier detection problem. 
To do so, we consider the samples with label 2 to be the inliers, and the sample with label 4 to be the outliers.

First, let us import the dataset, separate the features ``X`` and the target ``y`` and 
recode the inliers (``y=0``) and outliers (``y=1``). 

```python
import numpy as np
from icspylab import ICS, med_crit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import fetch_covtype
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
```

```python
X, y = fetch_covtype(return_X_y=True, as_frame=True)
s = (y == 2) + (y == 4)
X = X.loc[s]
y = y.loc[s]
y = (y != 2).astype(np.int32)

mask = X.notna().all(axis=1) & y.notna()
X = X.loc[mask]
y = y.loc[mask]

print("X shape:", X.shape)
```

```text
X shape: (286048, 54)
```

The features contain a lot of dummy variables ``Soil_Type``. 
The exploration of the features reveals that many of them contains almost only 0 values, 
which can cause collinearity issues for ICS. We decide to drop them to avoid any issues. 

```python
# Features cleaning
zero_ratio = (X == 0).mean()
cols_to_drop = zero_ratio[zero_ratio > 0.999].index
print("Features with more than 99.9% of 0 values:\n", cols_to_drop)
X = X.drop(cols_to_drop, axis=1)
print("X shape:", X.shape)
```

```text
Features to drop (more than 99.9% of 0 values):
 Index(['Soil_Type_0', 'Soil_Type_4', 'Soil_Type_6', 'Soil_Type_7',
       'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_20', 'Soil_Type_34',
       'Soil_Type_35', 'Soil_Type_36'],
      dtype='object')
X shape: (286048, 44)
```

As the dataset contains a lot of observations, we select 5% of the samples 
for training and 5% for testing.

```python
# Train test split
X_train, X_other, y_train, y_other = train_test_split(X, y, train_size=0.05, stratify=y, random_state=42)
X_test, _, y_test, _ = train_test_split(X_other, y_other, train_size=0.05, stratify=y_other, random_state=42)

n_samples, n_features = X_train.shape
anomaly_frac = y_train.mean()
print(f"{n_samples} datapoints with {y_train.sum()} anomalies ({anomaly_frac:.02%}) and {n_features} features")
```

```text
14302 datapoints with 137 anomalies (0.96%) and 44 features
```

Alright, we have 0.96% of anomalies. Let's see how LOF is performing to identify 
them and if the performance improve if we apply LOF on the invariant coordinates 
instead of the original features. 

The first pipeline is without ICS. LOF requires the data to be standardized so the pipeline 
consists in applying a [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler),
particularly suited in the presence of outliers, as a preprocessing step and then applying 
LOF on the standardized data. 

```python
lof_plain = make_pipeline(
    RobustScaler(),
    LocalOutlierFactor(
        n_neighbors=int(n_samples * anomaly_frac),
        novelty=True)
)

lof_plain.fit(X_train)
scores_plain = -lof_plain.decision_function(X_test)
```

The second pipeline includes another preprocessing step before the standardization. It calls
ICS to compute the invariant components and reduce the dimension with the med_crit criterion. 

```python
lof_ics  = make_pipeline(
    ICS(method_select=med_crit),
    RobustScaler(),
    LocalOutlierFactor(
        n_neighbors=int(n_samples * anomaly_frac),
        novelty=True)
)

lof_ics.fit(X_train)
scores_ics = -lof_ics.decision_function(X_test)
```

We compare the two methods with ROC curves, confusion matrices and F1 score. 

```python
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
```

```python
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
```

```python
# F1 scores
f1_ics = f1_score(y_test, y_pred_ics_bin)
f1_plain = f1_score(y_test, y_pred_plain_bin)

print(f"F1 score ICS + LOF: {f1_ics:.3f}")
print(f"F1 score LOF only: {f1_plain:.3f}")
```

```python
```