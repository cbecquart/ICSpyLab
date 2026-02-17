# Outlier detection

This example illustrates how ICS can be a relevant preprocessing step for an 
outlier detection problem. Indeed, ICS concentrates non-Gaussian structure into a reduced set of invariant 
coordinates, which can then be exploited by standard outlier detection algorithms.
This example also shows how easy it is to integrate ICS in a
scikit learn pipeline. 

The outlier detection algorithms we will consider are [Local Outlier Factor](https://scikit-learn.org/stable/modules/outlier_detection.html#local-outlier-factor) 
(LOF) and [Isolation Forest](https://scikit-learn.org/stable/modules/outlier_detection.html#isolation-forest)
(IForest), both implemented in scikit-learn. 

The problem we consider comes from the scikit-learn’'s [Evaluation of outlier detection estimators](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_outlier_detection_bench.html#forest-covertypes-dataset) 
example.
The dataset is the [Forest covertypes](https://scikit-learn.org/stable/datasets/real_world.html#covtype-dataset) 
dataset. It contains patches of forest and the target is the dominant species of tree in the patch.
There are 54 features whose description is available online. 
The prediction of the target variable is a multiclass classification problem, as there are 7
covertypes. 
However, following scikit-learn's [Evaluation of outlier detection estimators](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_outlier_detection_bench.html#forest-covertypes-dataset) 
example, we will adapt the dataset to make it an outlier detection problem. 
To do so, we consider the samples with label 2 to be the inliers, and the sample with label 4 to be the outliers.

First, let us import the dataset, separate the features ``X`` and the target ``y`` and 
encode the inliers (``y=0``) and outliers (``y=1``) to follow scikit-learn's evaluation conventions.

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
The exploration of the features reveals that many of them contains almost only zeros.
Such features may lead to nearly singular scatter matrices, which can affect the eigen decomposition underlying ICS.
We decide to drop them to avoid any issues. 

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

As the dataset contains over 280,000 samples, we have subsampled it to reduce the computational cost 
while maintaining class imbalance. We select 5% of the samples 
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

The training set contains approximately 0.96% anomalies, reflecting a highly imbalanced detection problem. 
Let's see how LOF and IForest are performing to identify 
them and if their performance improves if we apply them on the invariant coordinates 
instead of the original features. 

## LOF

The first pipeline is LOF without ICS. LOF requires the data to be standardized so the pipeline 
consists in applying a [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler),
particularly suited in the presence of outliers, as a preprocessing step and then applying 
LOF on the standardized data.
Following the scikit-learn tutorial, we set the number of neighbours proportionally to the estimated 
contamination rate. This ensures that the local density comparison reflects the expected proportion 
of anomalies.

```python
# LOF

lof_plain = make_pipeline(
    RobustScaler(),
    LocalOutlierFactor(
        n_neighbors=int(n_samples * anomaly_frac),
        novelty=True)
)

lof_plain.fit(X_train)
scores_lof_plain = -lof_plain.decision_function(X_test)
# Predictions
y_pred_plain = lof_plain.predict(X_test)
y_pred_plain_bin = (y_pred_plain == -1).astype(int)
```

The second pipeline includes another preprocessing step before the standardization. It calls
ICS to compute the invariant components and reduce the dimension with the ``med_crit`` criterion:
it selects invariant components associated with extreme generalized kurtosis values.
ICS is affine invariant, but we still apply scaling afterward to ensure that the downstream LOF 
algorithm operates on standardized invariant coordinates.

```python
# LOF with ICS

lof_ics  = make_pipeline(
    ICS(method_select=med_crit),
    RobustScaler(),
    LocalOutlierFactor(
        n_neighbors=int(n_samples * anomaly_frac),
        novelty=True)
)

lof_ics.fit(X_train)
scores_lof_ics = -lof_ics.decision_function(X_test)
# Predictions
y_pred_ics = lof_ics.predict(X_test)      # 1=inlier, -1=outlier
y_pred_ics_bin = (y_pred_ics == -1).astype(int)  # 1=outlier, 0=inlier
```

## IForest

Now, let's do the same for Isolation Forest. This algorithm does not need to standardize the data and we will keep
the default parameters. 

```python
# IForest

if_plain = IsolationForest(random_state=42)

if_plain.fit(X_train)
scores_if_plain = -if_plain.decision_function(X_test)
# Predictions
y_pred_if_plain = if_plain.predict(X_test)
y_pred_if_plain_bin = (y_pred_if_plain == -1).astype(int)
```

Finally, we cretae a pipeline performing ICS and then applying IForest to the selected invariant components, 
selected via the ``med_crit`` criterion. 

```python
# IForest with ICS

if_ics = make_pipeline(
    ICS(method_select=med_crit),
    IsolationForest(random_state=42)
)

if_ics.fit(X_train)
scores_if_ics = -if_ics.decision_function(X_test)
# Predictions
y_pred_if_ics = if_ics.predict(X_test)      # 1=inlier, -1=outlier
y_pred_if_ics_bin = (y_pred_if_ics == -1).astype(int)  # 1=outlier, 0=inlier
```

## Results

We compare the four methods with ROC curves, confusion matrices and F1 scores. 

```python
# ROC curves

fpr_lof_ics, tpr_lof_ics, _ = roc_curve(y_test, scores_lof_ics)
auc_lof_ics = auc(fpr_lof_ics, tpr_lof_ics)

fpr_lof_plain, tpr_lof_plain, _ = roc_curve(y_test, scores_lof_plain)
auc_lof_plain = auc(fpr_lof_plain, tpr_lof_plain)

fpr_if_ics, tpr_if_ics, _ = roc_curve(y_test, scores_if_ics)
auc_if_ics = auc(fpr_if_ics, tpr_if_ics)

fpr_if_plain, tpr_if_plain, _ = roc_curve(y_test, scores_if_plain)
auc_if_plain = auc(fpr_if_plain, tpr_if_plain)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Subplot 1 : LOF
axes[0].plot(fpr_lof_ics, tpr_lof_ics, label=f"ICS + LOF (AUC = {auc_lof_ics:.3f})")
axes[0].plot(fpr_lof_plain, tpr_lof_plain, label=f"LOF only (AUC = {auc_lof_plain:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", label="Random")

axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve – LOF")
axes[0].legend()
axes[0].grid(True)

# Subplot 2 : Isolation Forest
axes[1].plot(fpr_if_ics, tpr_if_ics, label=f"ICS + IF (AUC = {auc_if_ics:.3f})")
axes[1].plot(fpr_if_plain, tpr_if_plain, label=f"IF only (AUC = {auc_if_plain:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", label="Random")

axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve – Isolation Forest")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

```{image} ../_static/outliers_ROC.png
:alt: ROC curves of the outliers example
:width: 700px
:align: center
```

```python
# Confusion matrices

cm_lof_ics = confusion_matrix(y_test, y_pred_ics_bin)
cm_lof_plain = confusion_matrix(y_test, y_pred_plain_bin)

cm_if_ics = confusion_matrix(y_test, y_pred_if_ics_bin)
cm_if_plain = confusion_matrix(y_test, y_pred_if_plain_bin)


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ConfusionMatrixDisplay(cm_lof_ics, display_labels=["Inlier", "Outlier"]).plot(
    ax=axes[0, 0], cmap=plt.cm.Blues, colorbar=False
)
axes[0, 0].set_title("ICS + LOF")

ConfusionMatrixDisplay(cm_lof_plain, display_labels=["Inlier", "Outlier"]).plot(
    ax=axes[1, 0], cmap=plt.cm.Oranges, colorbar=False
)
axes[1, 0].set_title("LOF only")

ConfusionMatrixDisplay(cm_if_ics, display_labels=["Inlier", "Outlier"]).plot(
    ax=axes[0, 1], cmap=plt.cm.Blues, colorbar=False
)
axes[0, 1].set_title("ICS + IF")

ConfusionMatrixDisplay(cm_if_plain, display_labels=["Inlier", "Outlier"]).plot(
    ax=axes[1, 1], cmap=plt.cm.Oranges, colorbar=False
)
axes[1, 1].set_title("IF only")

plt.tight_layout()
plt.show()
```

```{image} ../_static/outliers_CM.png
:alt: Confusion matrices of the outliers example
:width: 600px
:align: center
```

```python
# F1 scores

f1_plain = f1_score(y_test, y_pred_plain_bin)
f1_ics = f1_score(y_test, y_pred_ics_bin)

print(f"F1 score LOF only: {f1_plain:.3f}")
print(f"F1 score ICS + LOF: {f1_ics:.3f}")

f1_if_plain = f1_score(y_test, y_pred_if_plain_bin)
f1_if_ics = f1_score(y_test, y_pred_if_ics_bin)

print(f"F1 score IF only: {f1_if_plain:.3f}")
print(f"F1 score ICS + IF: {f1_if_ics:.3f}")
```

```text
F1 score LOF only: 0.010
F1 score ICS + LOF: 0.160
F1 score IF only: 0.050
F1 score ICS + IF: 0.481
```

Adding ICS as a pre-processing step improves the area under the curve (AUC) for both LOF and Isolation 
Forest analysis. However, given the strong class imbalance, this metric is not very informative. 
In contrast, the increase in the F1 score is more meaningful: it rises from 0.010 to 0.160 for LOF and 
from 0.050 to 0.481 for Isolation Forest. 
The confusion matrices further demonstrate that ICS helps to detect many more anomalies while only moderately 
increasing the number of false positives, particularly in the case of Isolation Forest. 

Overall, ICS clearly enhances the performance of anomaly detection on imbalanced datasets.
