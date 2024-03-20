from abc import ABC, abstractmethod

import numpy as np
from scipy import stats

__doc__ = """Samplers for 1 and higher dimensional problems.

For the Latin Hypercube Designs, consider the Gramacy (2020) book _Surrogates_
https://bobby.gramacy.com/surrogates/. We generalize the concept by using
hypercubes other than [0,1]^n in our `LatinHypercubeSampling` strategy (see the
implementation for details).
"""


class SamplingStrategy(ABC):
    n_dimensions: int

    @abstractmethod
    def sample(*args, **kwargs):
        """Base method for sampling from a sampling strategy."""
        ...


class FactorialSampling(SamplingStrategy):
    """Sampling using a factorial design to draw new covariate values."""

    def __init__(self, *factors: np.ndarray):
        self.factors = factors

    def sample(self):
        """Draws a factorial design given sample values to use in each dimension."""
        grid = np.meshgrid(*self.factors)
        return np.hstack([g.reshape(-1, 1, order="F") for g in grid])


class RandomSampling(SamplingStrategy):
    """Random sampling from a prior distribution for the covariates."""

    def __init__(self, *distributions: stats.rv_continuous):
        self.distributions = distributions

    def sample(self, n_samples: int):
        ndims = len(self.distributions)
        samples = np.empty((ndims, n_samples))
        for i, distribution in enumerate(self.distributions):
            samples[i] = distribution.rvs(size=n_samples)
        return samples.T


class LatinHypercubeSampling(SamplingStrategy):
    """Sampling using a Latin hypercube design to draw new covariate values."""

    def __init__(
        self, *, input_dims: list[tuple[float, float]] | None = None, n_dimensions=2
    ):
        self.n_dimensions = n_dimensions
        self.input_dims = input_dims

    def __call__(self, n_samples):
        return self.sample(n_samples)

    def sample(self, n_samples: int):
        if self.input_dims is None:
            intervals = np.linspace(0, 1, n_samples + 1)
        else:
            intervals = np.array(
                [np.linspace(*dim, n_samples + 1) for dim in self.input_dims]
            )

        random_values = np.random.uniform(
            intervals[..., :-1], intervals[..., 1:], (self.n_dimensions, n_samples)
        )

        for dimension_values in random_values:
            np.random.shuffle(dimension_values)

        return random_values.T


def main():
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    ls = np.array([0.25, 0.5, 0.75])
    smooth = np.linspace(-2, 2, 20)
    fac = FactorialSampling(ls, smooth)
    fac_samples = fac.sample()
    print(fac_samples.shape)
    plt.scatter(fac_samples[0], fac_samples[1])
    plt.title("Factorial sampling design")
    plt.xlabel("Spatial range (lengthscale)")
    plt.ylabel("Smoothness")
    plt.savefig("fac.png")

    rand_sample = RandomSampling(stats.uniform(), stats.norm())
    rand_samples = rand_sample.sample(100)
    print(rand_samples.shape)
    sns.pairplot(pd.DataFrame(rand_samples))
    plt.savefig("rand.png")
    plt.close()

    lhs = LatinHypercubeSampling(input_dims=[(0, 1), (0.25, 3.25)])
    lhs_samples = lhs.sample(250)
    print(lhs_samples.shape)

    plt.scatter(lhs_samples[0], lhs_samples[1])
    plt.savefig("lhs.png")
    plt.close()

    lhs_samples = pd.DataFrame(lhs_samples)
    sns.pairplot(lhs_samples)
    plt.savefig("lhs_pairplot.png")
    plt.close()


if __name__ == "__main__":
    main()
