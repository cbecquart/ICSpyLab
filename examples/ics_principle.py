import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from math import atan2
from scipy.stats import chi2
from icspylab.scatter import cov, cov4
from icspylab.ics import ICS


path = "2_viz_output/"

def create_ellipse_points(loc, cov):
    """
    Creates points to draw the ellipse of a distribution.

    Parameters:
        loc (array-like of shape (2,): location parameter
        cov (array-like of shape (2,2): scatter matrix

    Returns:
        ndarray, ndarray: x and y coordinates of points to draw the ellipse
    """

    dist = np.sqrt(chi2.ppf(0.975, 2))

    # Compute eigenvectors and eigenvalues
    cov_eigenval, cov_eigenvect = np.linalg.eig(cov)
    lambda1 = np.max(cov_eigenval)
    lambda2 = np.min(cov_eigenval)
    cov_eigenvect_max = cov_eigenvect[:, np.argsort(cov_eigenval)[1]]

    # Compute the angle between the x-axis and the largest eigenvector, output must be between 0 and 2pi
    alfa = atan2(cov_eigenvect_max[1], cov_eigenvect_max[0])
    if alfa < 0:
        alfa += (2 * np.pi)

    # Create ellipse points in x and y coordinates
    z = np.arange(0, 2*np.pi, 0.01)
    z1 = dist * np.sqrt(lambda1) * np.cos(z)
    z2 = dist * np.sqrt(lambda2) * np.sin(z)

    # Define a rotation matrix and rotate the ellipse to angle alpha
    r = np.array([[np.cos(alfa), np.sin(alfa)], [-np.sin(alfa), np.cos(alfa)]])
    ellipse_points = loc + np.dot(np.column_stack((z1, z2)), r)

    return ellipse_points[:, 0], ellipse_points[:, 1]


# Generate data

# Set parameters to generate data
n_samples = 980
n_outliers = 20
n_features = 2
add_outliers = True

# Generate Gaussian data of shape (125,2)
gen_cov = np.diag((1, 5))
gen_loc = np.array([0, 0])

X = np.random.default_rng().multivariate_normal(gen_loc, gen_cov, n_samples)
df = pd.DataFrame(X, columns=['X1', 'X2'])
df['Group'] = 'Group_1'

# Add some outliers
if add_outliers:
    outlier_cov = gen_cov
    outlier_loc = np.array([8, -2])
    X_out = np.random.default_rng().multivariate_normal(outlier_loc, outlier_cov, n_outliers)
    X = np.concatenate((X, X_out), axis=0)
    df_out = pd.DataFrame(X_out, columns=['X1', 'X2'])
    df_out['Group'] = 'Group_2'
    df = pd.concat([df, df_out])


# Scatter matrices

# Compute two scatters
cov_X = cov(X)
cov4_X = cov4(X)

# Create ellipses
x_emp, y_emp = create_ellipse_points(cov_X.location, cov_X.scatter)
x_cov4, y_cov4 = create_ellipse_points(cov4_X.location, cov4_X.scatter)

# Plot groups
fig = px.scatter(df, x='X1', y='X2', color='Group', color_discrete_sequence=['rgb(31, 119, 180)', 'rgb(255, 127, 14)'])
# fig.update_yaxes(range=[-4, 4])
# fig.update_xaxes(range=[-10, 13])
fig.update_xaxes(scaleanchor='y')
fig.update_yaxes(scaleanchor='x')
fig.update_yaxes(range=[-10, 9])
fig.update_xaxes(range=[-10, 13])
fig.update_layout(width=600, height=525,
                  margin=dict(l=20, r=20, t=20, b=20)
                  )
fig.write_image('../docs/_static/ellipses_empty.png')

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(solver='eigen', n_components=1)
X_lda = lda.fit_transform(X, df["Group"])
lda_coef = lda.scalings_[:, 0]

origin = np.mean(X, axis=0)
axis_length = 20
axis_start = origin - axis_length * lda_coef
axis_end = origin + axis_length * lda_coef
fig.add_trace(go.Scatter(x=[axis_start[0], axis_end[0]], y=[axis_start[1], axis_end[1]],
                         line_color='rgb(23, 190, 207)', mode='lines', name='LDA'))
fig.update_layout(width=600, height=525,
                  margin=dict(l=20, r=20, t=20, b=20)
                  )
fig.write_image('../docs/_static/ellipses_lda.png')

# Add ellipses
fig.add_scatter(x=x_emp, y=y_emp, mode='lines', name='COV', line_color='rgb(188, 189, 34)')
fig.add_scatter(x=x_cov4, y=y_cov4, mode='lines', name='COV4', line_color='rgb(148, 103, 189)')
fig.update_layout(width=600, height=525,
                  margin=dict(l=20, r=20, t=20, b=20)
                  )
fig.write_image('../docs/_static/ellipses_covcov4.png')

# Add PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA()
pca.fit(X) # or X_scaled
first_component = pca.components_[0]
variance = pca.explained_variance_[0]
axis_start = pca.mean_ - (axis_length-5) * first_component
axis_end = pca.mean_ + (axis_length-5) * first_component
fig.add_trace(go.Scatter(x=[axis_start[0], axis_end[0]], y=[axis_start[1], axis_end[1]],
                         line_color='rgb(253, 108, 158)', mode='lines', name='PC1'))

# ICS
ics = ICS(S1=cov, S2=cov4, algorithm="whiten")
X_new = ics.fit_transform(X)

ics_coef = ics.W_
axis_start = origin - axis_length * ics_coef[0]
axis_end = origin + axis_length * ics_coef[0]
fig.add_trace(go.Scatter(x=[axis_start[0], axis_end[0]], y=[axis_start[1], axis_end[1]],
                         line_color='rgb(214, 39, 40)', mode='lines', name='ICS'))

fig.update_layout(width=600, height=525,
                  margin=dict(l=20, r=20, t=20, b=20)
                  )
fig.show()
fig.write_image('../docs/_static/ellipses.png')
