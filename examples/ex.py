import numpy as np
from sklearn.datasets import load_iris
from icspylab import ICS, normal_crit, median_crit, unimodal_crit, plot_ics, dftu
from icspylab.scatter import cov, cov4, covAxis, covW, mcd, tM, tcov, tcovAxis
from icspylab.distributions import *

# X_1d = np.random.randn(10)
# s = tcovAxis(X_1d)
# print(s)

eps = [0.5, 0.5]
mu = [np.ones(2), np.ones(2)*10]
sigma = [np.eye(2) for _ in range(2)]

X, labels = generate_gaussian_mixture(eps, mu, sigma, n=1000, p=6)
plot_ics(X)



X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
cov_X = cov(X)
print(cov_X.scatter)

cov4_X = cov4(X)
print(cov4_X.scatter)

covAxis_X = covAxis(X)
print(covAxis_X.scatter)

covW_X = covW(X)
print(covW_X.scatter)

mcd_X = mcd(X, support_fraction=None)
print(mcd_X.scatter)

tM_X = tM(X)
print(tM_X.scatter)

tcov_X = tcov(X)
print(tcov_X.scatter)

tcovAxis_X = tcovAxis(X)
print(tcovAxis_X.scatter)


iris = load_iris()
X = iris.data

stat, p = dftu(X[:,0])
print(round(stat, 2), round(p, 2))

ics = ICS(S1="cov", S2="cov4")
X_new = ics.fit_transform(X)

plot_ics(X_new)

selection_res = normal_crit(X=X, W=ics.components_)
print(selection_res.info)

selection_res = median_crit(kurtosis=ics.kurtosis_, W=ics.components_)
print(selection_res.info)

selection_res = unimodal_crit(X=X, W=ics.components_)
print(selection_res.info)


# selection_res = normal_crit(ics.scores_, level=0.05,
#             test="agostino_test",
#             max_select=None)
# comp_names = [f"IC_{i + 1}" for i in range(X.shape[1])]
# name_to_idx = {name: i for i, name in enumerate(comp_names)}
# idx = [name_to_idx[name] for name in selection_res["select"]]
# X_new = X[:, idx]
# print(X_new)
#
# a = normal_crit(ics.scores_, level=0.05,
#             test="agostino_test",
#             max_select=None)
# print(a)
#
# a = normal_crit(ics.scores_, level=0.05,
#             test="agostino_test",
#             max_select=None)
# print(a)
#
# a = normal_crit(ics.scores_, level=0.05,
#             test="agostino_test",
#             max_select=None)
# print(a)
#
# a = normal_crit(ics.scores_, level=0.05,
#             test="agostino_test",
#             max_select=None)
# print(a)

