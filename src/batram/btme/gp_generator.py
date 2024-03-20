import numpy as np
from scipy import linalg, stats
from sklearn.gaussian_process import kernels

__doc__ = """A Gaussian process generator helper.

This module contains a helper class for generating Gaussian process data, and
a helper function for making a grid of locations in a unit hypercube.

Example:
    >>> locs = make_grid(32, 2)
    >>> kernel = kernels.Matern(nu=1.5, length_scale=0.25)
    >>> gp = GaussianProcessGenerator(locs, kernel, 0.01)
    >>> gp_samples = gp.sample(7)
"""


def make_grid(nlocs: int, ndims: int) -> np.ndarray:
    """Make a grid of equally spaced points in a unit hypercube.

    Args:
        nlocs: The number of locations in each dimension.
        ndims: The number of dimensions.

    Returns:
        A numpy array of shape (nlocs**ndims, ndims) containing the locations
        of the data points.
    """
    _ = np.linspace(0, 1, nlocs)
    return np.stack(np.meshgrid(*[_] * ndims), axis=-1).reshape(-1, ndims)


class GaussianProcessGenerator:
    """Numpy-based Gausssian Process data generator.

    Attributes:
        locs: A numpy array of shape (nlocs, ndims) containing the locations
            of the data points.
        kernel: A sklearn.gaussian_process.kernels.Kernel object.
        sd_noise: The standard deviation of the noise.
    """

    def __init__(self, locs: np.ndarray, kernel: kernels.Kernel, sd_noise: float):
        self.locs = locs
        self.kernel = kernel
        self.sd_noise = sd_noise

    def update_kernel(self, **params):
        self.kernel.set_params(**params)

    def sample(self, num_reps: int = 1):
        """Sample from the Gaussian Process."""
        cov = self.kernel(self.locs)
        cov += self.sd_noise**2 * np.eye(cov.shape[0])
        chol = linalg.cholesky(cov, lower=True)
        z = stats.multivariate_normal(cov=np.eye(chol.shape[-1])).rvs(num_reps)
        return np.dot(chol, z.T).T
