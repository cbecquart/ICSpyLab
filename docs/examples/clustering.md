# Clustering

In this example, ICS is used for exploratory data analysis.
Using a real-world dataset, it illustrates how ICS can reveal interesting structures 
in the data - in this case, clusters.

The dataset used here is the same as in the Outlier Detection tutorial, namely the 
[Forest covertypes](https://scikit-learn.org/stable/datasets/real_world.html#covtype-dataset) 
dataset.
It contains cartographic information about forest patches, where the target variable 
corresponds to the dominant tree species in each patch. 
The dataset includes 54 features whose description is available online.

This example focuses on exploratory analysis of the observations belonging to cover type 2, which is 
the largest class in the dataset.

First, let us load the dataset, separate the features X and the target y, 
and filter the observations corresponding to cover type 2 (``y=2``).

```python
import numpy as np
from sklearn.datasets import fetch_covtype

X, y = fetch_covtype(return_X_y=True, as_frame=True)
print(y.value_counts())
s = (y == 2)
X = X.loc[s]
y = y.loc[s]

print("X shape:", X.shape)
```

```text
Cover_Type
2    283301
1    211840
3     35754
7     20510
6     17367
5      9493
4      2747
Name: count, dtype: int64

X shape: (283301, 54)
```

As in the Outlier Detection example, we remove variables containing 
almost only zeros in order to avoid singularity issues.

```python
# Features cleaning
zero_ratio = (X == 0).mean()
cols_to_drop = zero_ratio[zero_ratio > 0.999].index
print("Features to drop (more than 99.9% of 0 values):\n", cols_to_drop)
X = X.drop(cols_to_drop, axis=1)
print("X shape:", X.shape)
```

Since the dataset contains more than 280,000 observations, we keep only 5% of 
the data to reduce the computational cost.

```python
from sklearn.model_selection import train_test_split

# Subsample the data
X_sub, _, y_train, _ = train_test_split(X, y, train_size=0.05, stratify=y, random_state=42)
print("X_sub shape:", X_sub.shape)
```

## PCA

```{image} ../_static/clustering_pca.PNG
:alt: PCA results of the clustering example
:width: 700px
:align: center
```

## ICS

```{image} ../_static/clustering_ics.PNG
:alt: ICS results of the clustering example
:width: 700px
:align: center
```
