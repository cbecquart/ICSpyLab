from sklearn.datasets import load_iris
from sklearn.covariance import MinCovDet
from icspylab import ICS, Scatter

# Load dataset
iris = load_iris()
X = iris.data

# Compute MCD scatter (covariance) and location
mcd_fit = MinCovDet().fit(X)
mcd_loc = mcd_fit.location_
mcd_cov = mcd_fit.covariance_

print("location: ", mcd_loc)
print("scatter: ", mcd_cov)


# Define a function returning a Scatter object
def mcd_scatter(X, **kwargs):
    mcd_fit = MinCovDet(**kwargs).fit(X)
    mcd_loc = mcd_fit.location_
    mcd_cov = mcd_fit.covariance_

    return Scatter(location=mcd_loc, scatter=mcd_cov, label="MCD")

print(mcd_scatter(X))

# Instantiate ICS with the MCD as S1
ics = ICS(S1=mcd_scatter, S2="cov")

# Fit and transform the ICS model
X_new = ics.fit_transform(X)
