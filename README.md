# ICSpyLab

## Overview

Invariant Coordinate Selection (ICS) is a data transformation method. It transforms the data, via the simultaneous
diagonalization of two scatter matrices, into an invariant coordinate system or independent components,
depending on the underlying assumptions.
It is particularly useful for dimension reduction. Unlike PCA, ICS is not based on variance maximization but on the
maximization/minimization of a generalized kurtosis, and it is invariant not only to orthogonal data transformations but
to any affine transformation.

This package brings the main functionalities of the [ICS R package](https://cran.r-project.org/web/packages/ICS/index.html)
to Python, offering tools for identifying and selecting invariant coordinates in multivariate data.
It includes various covariance estimators, transformation settings,
and plotting utilities. Our extensive testing ensures results consistent with the R package, making it easy for users to
transition from R to Python or start fresh with ICS.

Check out the [documentation](https://icspylab.readthedocs.io/en/latest/) for more details.

## Installation

```bash
pip install numpy pandas scipy scikit-learn seaborn matplotlib 
pip install ics
```

### Usage

```python
from icspylab import ICS, cov, covW
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data

# Instantiate ICS object
# ics = ICS() # default parameters
ics = ICS(S1=cov, S2=covW, algorithm='standard', S2_args={'alpha': 1, 'cf': 2})

# Fit and transform the ICS model (equivalent of the function ICS-S3() from the R package ICS)
ics.fit_transform(X)

# Printing a summary
ics.describe()
```
