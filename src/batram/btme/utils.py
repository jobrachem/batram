import numpy as np
import torch

from batram.datautils import Dataset


def get_locs(lat: np.ndarray, lon: np.ndarray):
    """Gets locations as a flattened grid from lat and lon coordinate arrays.

    This function makes a flattened array of locations from the lat and lon
    coordinate arrays so we can sort the spatial fields. Care should be taken
    to ensure the flattened locations have the same orientation as the flattened
    field arrays. This can be done by plotting the original fields using
    `imshow` and then plotting the flattened fields using `scatter` with the
    flattened locations as the `x` and `y` arguments.

    Args:
        lat: 1d array of latitudes.
        lon: 1d array of longitudes.

    Returns:
        locs: 2d array of locations.
    """
    locs = np.stack(
        [
            np.repeat(lon, lat.size),
            np.tile(np.flip(lat), lon.size),
        ],
        axis=-1,
    )
    return locs


def train_test_split(
    dataset: Dataset, train_size: float = 0.9, seed: int = 42
) -> tuple[Dataset, Dataset]:
    """Splits a dataset into a training and testing dataset.

    Args:
        dataset: Dataset to split.
        train_size: Fraction of the dataset to use for training.

    Returns:
        Tuple of training and testing datasets.
    """
    np.random.seed(seed)
    n = dataset.response.shape[1]
    num_train = int(train_size * n)
    indices = np.random.permutation(n)

    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    assert dataset.x is not None

    train_dataset = Dataset(
        dataset.locs,
        dataset.response[:, train_indices, :].squeeze().mT,
        dataset.condsets,
        dataset.x[:, train_indices, :],
    )
    test_dataset = Dataset(
        dataset.locs,
        dataset.response[:, test_indices, :].squeeze().mT,
        dataset.condsets,
        dataset.x[:, test_indices, :],
    )

    return train_dataset, test_dataset


def normalize_input(x: torch.Tensor) -> torch.Tensor:
    """Normalizes 3D tensor x over the sample dimension (index 1)."""

    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)


def get_dataset(
    locs: torch.Tensor,
    condsets: torch.Tensor,
    response: torch.Tensor,
    x: torch.Tensor,
    **device_kwargs,
) -> Dataset:
    x = normalize_input(x)
    return Dataset(
        locs.to(**device_kwargs),
        condsets.to(**device_kwargs),
        response.to(**device_kwargs),
        x.to(**device_kwargs),
    )


def to_device(dataset, device):
    attrs = ["response", "augmented_response", "condsets", "x", "scales"]
    for attr in attrs:
        setattr(dataset, attr, getattr(dataset, attr).to(device=device))
        setattr(dataset, attr, getattr(dataset, attr).to(device=device))
