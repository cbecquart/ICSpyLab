import numpy as np
from sklearn.datasets import load_iris
from icspylab import ICS, normal_crit, med_crit, plot_ics
from icspylab.scatter import cov, cov4, covAxis, covW, mcd, tM, tcov, tcov2

# X_1d = np.random.randn(10)
# s = tcov2(X_1d)
# print(s)


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

tcov2_X = tcov2(X)
print(tcov2_X.scatter)


iris = load_iris()
X = iris.data

ics = ICS()
X_new = ics.fit_transform(X)
plot_ics(X_new)

ics = ICS(S1="cov", S2="cov4")
X_new = ics.fit_transform(X)

selection_res = normal_crit(X=X_new, W=ics.components_)
print(selection_res.info)

selection_res = med_crit(kurtosis=ics.kurtosis_, W=ics.components_)
print(selection_res.info)

a=0
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

