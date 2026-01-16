import numpy as np
import matplotlib.pyplot as plt

def generate_points_on_random_ball(n, p, radius=1):
    """
    Generate random points in d dimensions that have uniform probability over the unit ball scaled by "radius" (length
    of points are in range [0, "radius"]).
    Source: https://stackoverflow.com/questions/54544971/how-to-generate-uniform-random-points-inside-d-dimension-ball-sphere
    :param n: (int) number of points to generate
    :param p: (int) dimension
    :param radius: (float) radius of the d-ball
    :return: (n, d) array with the generated points
    """
    # Distribute evenly on ball by normalizing the length of a vector of random-normal values
    random_directions = np.random.normal(size=(p, n))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    # Generate a random radius with probability proportional to the surface area of a ball with a given radius
    # random_radii = np.random.random(n) ** (1 / p)
    # Return the list of random (direction & length) points.
    return radius * (random_directions).T

def points_uniformes_sphere(n, p, rayon=np.sqrt(2)):
    x = np.random.normal(size=(n, p))
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return rayon * x / norms


p = 4
sigma = 0.2
n = 500

rng = np.random.default_rng(seed=0)

signal = points_uniformes_sphere(n, 2, rayon=np.sqrt(2))

X1 = signal + rng.multivariate_normal(
    mean=np.zeros(2),
    cov=sigma**2 * np.eye(2),
    size=n
)

X2 = rng.multivariate_normal(
    mean=np.zeros(p-2),
    cov=(1 + sigma**2) * np.eye(p-2),
    size=n
)

X = np.hstack((X1, X2))

plt.scatter(X[:, 0], X[:, 1])
plt.show()
