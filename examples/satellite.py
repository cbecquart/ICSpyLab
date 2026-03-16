import numpy as np
import pandas as pd
import plotly.express as px
from icspylab import ICS, cov, covW, cov4, mcd, tcov, tcovAxis
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM


X = pd.read_csv("satellite-unsupervised-ad.csv", header=None,
                names=[f'C_{i+1}' for i in range(37)])

labels = X["C_37"].to_numpy()
X.drop("C_37", axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

ics = ICS(S1=cov, S2=cov4, algorithm='standard')
X_train_new = ics.fit_transform(X_train)
X_test_new = ics.transform(X_test)

fig = px.scatter(x=X_train_new[:,0], y=X_train_new[:,1], color=y_train)
fig.show()

pipe = Pipeline([
    ("ics", ICS(algorithm="whiten")),
    ("svm", OneClassSVM(kernel="rbf", nu=0.05))
])

pipe.fit(X_train)

y_pred = pipe.predict(X_test)

fig = px.scatter(x=X_test_new[:,0], y=X_test_new[:,1], color=y_test, symbol=y_pred)
fig.show()

# param_grid = {
#     "ics__S1": [cov, mcd, tcov],
#
#     "svm__kernel": ["rbf"],
#     "svm__nu": [0.01, 0.05, 0.1],
#     "svm__gamma": ["scale", 0.1, 1],
# }
#
# grid = GridSearchCV(
#     pipe,
#     param_grid,
#     scoring=None,
#     cv=3
# )
#
# grid.fit(X_train, y_train)
#
# print("Best parameters:")
# print(grid.best_params_)
#
# best_model = grid.best_estimator_
# y_pred = best_model.predict(X_test)

