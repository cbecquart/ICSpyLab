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

Install the package using pip:
```bash
pip install icspylab
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

## Examples

All examples and figures shown in the documentation are reproducible and available as standalone Python 
scripts in the `examples/` directory.


## 🤝 Contributing

We welcome contributions of all kinds, including bug fixes, new features, and documentation improvements.

To get started, please check out our [CONTRIBUTING.md](CONTRIBUTING.md) guide.


## Testing

The default CI workflow executes the full testing suite on Linux and macOS, including validation against the reference R implementation.

On Windows, the CI workflow runs the Python test suite only, without the R-based comparison tests, due to 
platform-specific constraints in the testing environment.

## Citation

If you use this software, please cite:

Becquart, C. and Abdelsameia, A. (2026). ICSpyLab (Version 1.0.1)
https://doi.org/10.5281/zenodo.20707572

```bibtex
@software{becquart2026,
 author = {Becquart, Colombe and Abdelsameia, Abdallah},
 title = {ICSpyLab},
 year = {2026},
 version = {1.0.1},
 doi = {10.5281/zenodo.20707572}
}
```
