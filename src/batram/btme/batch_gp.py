import gpytorch
import torch
from tqdm.notebook import tqdm

__doc__ = """Batched gp models for preprocessing covariate-dependent data."""


class BatchGP(gpytorch.models.ExactGP):
    """Batched Exact GP models (no approximation or scaling tricks) with noise.

    Originally written by @wiep. Modified by @danjdrennan to use more
    reproducibly in cluster environments.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
    ):
        super().__init__(train_x, train_y, likelihood)
        batch_size = torch.Size([train_x.shape[0]])
        ard_num_dims = train_x.shape[-1]

        self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_size)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                ard_num_dims=ard_num_dims,
                batch_shape=batch_size,
                nu=1.5,
            ),
            batch_shape=batch_size,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) # type: ignore


def get_model(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[BatchGP, gpytorch.likelihoods.Likelihood]:
    """Get a batched exact GP model.

    Args:
        x: Training data.
        y: Training labels.

    Returns:
        A batched exact GP model.
    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        batch_shape=torch.Size([x.shape[-1]])
    )
    model = BatchGP(x, y, likelihood)
    return model, likelihood


def preprocess_data(x, y) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess data for use in a batched exact GP model.

    Loads data and reshapes it to be compatible with batched exact GP models.
    Also normalizes both variables.
    """
    y = torch.from_numpy(y).float()
    y.permute(-2, -1, 0).reshape(-1, y.shape[0])
    y = (y - y.mean(0, keepdim=True)) / y.std(0, keepdim=True)

    x = torch.from_numpy(x).float()
    x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True)
    x = x.repeat(y.shape[0], 1, 1)

    return x, y


def train_model(num_epochs: int, lr: float, model, likelihood, train_x, train_y):
    """Train a batched exact GP model.

    Args:
        num_epochs: Number of epochs to train for.
        lr: Learning rate.
        model: Batched exact GP model.
        likelihood: Likelihood for the model.
        train_x: Training data.
        train_y: Training labels.

    Returns:
        Trained batched exact GP model.
    """
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, 3e-7)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    with tqdm(range(num_epochs)) as pbar:
        for i in pbar:
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y).mean()  # type: ignore
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_description(f"loss = {loss.item():.4f}")

    return model
