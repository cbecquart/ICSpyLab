import numpy as np
import pandas as pd
import plotly.express as px
from icspylab import ICS, cov, covW, cov4, mcd, tcov, tcov2


X = pd.read_csv("satellite-unsupervised-ad.csv", header=None,
                names=[f'C_{i+1}' for i in range(37)])

labels = X["C_37"].to_numpy()
X.drop("C_37", axis=1, inplace=True)

ics = ICS(S1=cov, S2=tcov2, algorithm='standard')
ics.fit_transform(X)

fig = px.scatter(x=ics.scores_[:,-1], y=ics.scores_[:,-2], color=labels)
fig.show()
fig.write_image("fig1.png")

