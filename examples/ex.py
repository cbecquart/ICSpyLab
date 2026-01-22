import numpy as np
from sklearn.datasets import load_iris
from icspylab import ICS, cov, tcov, cov4
from icspylab.comp_select import normal_crit, med_crit

data = load_iris()
X = data.data

ics = ICS(S1=cov, S2=cov4)
X_new = ics.fit_transform(X)


selection_res = med_crit(ics.kurtosis_)
print(selection_res)


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

