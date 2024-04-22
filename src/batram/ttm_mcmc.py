import logging
import torch
import veccs.orderings
from pathlib import Path

import numpy as np
import dill as pickle
from jax import Array as JaxArray
from liesel_ptm import (
    PTMLocScale,
    PTMLocScalePredictions,
    VarInverseGamma,
    cache_results,
)
import liesel.goose as gs
from numpy.typing import NDArray as NumpyArray
from scipy import stats

from .legmods import Data, SimpleTM
from batram.stopper import EarlyStopper

ArrayLike = JaxArray | NumpyArray

logger = logging.getLogger(__name__)


def setup_ptm(
    i: int,
    y: ArrayLike,
    nparam: int,
    tau2_cls: type | None = None,
    knots: ArrayLike | None = None,
    **tau2_kwargs,
):
    if tau2_cls is None:
        tau2 = VarInverseGamma(1.0, concentration=1.0, scale=0.5, name=f"tau2_{i}")
    else:
        tau2 = tau2_cls(
            **tau2_kwargs.get("tau2_kwargs"), name=f"tau2_{i}"
        )  # type: ignore

    if knots is None:
        model: PTMLocScale = PTMLocScale.from_nparam(
            y=y, nparam=nparam, normalization_tau2=tau2
        )
        model, _ = model.optimize_knots(
            optimize_params=[], knot_prob_levels=(0.01, 0.99)
        )
    else:
        model = PTMLocScale(knots, y, normalization_tau2=tau2)

    return model


def preprocess_transport_map_data(obs: ArrayLike, locs: ArrayLike, ntrain1: int, ntrain2: int, ntest: int) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    obs = torch.as_tensor(obs, dtype=torch.float32)
    ord = np.lexsort((locs[:, 1], locs[:, 0]))
    locs = locs[ord]
    obs = obs[:, ord]

    # Maximin ordering of the locations using the `veccs` package.
    # Note, the locations array is reordered over its first dimension, whereas the
    # observations are reordered over the last dimension.
    order = veccs.orderings.maxmin_cpp(locs)
    locs = locs[order, ...]
    obs = obs[..., order]

    train1 = obs[:ntrain1, :]
    train2 = obs[ntrain1:(ntrain1 + ntrain2), :]
    test = obs[-ntest:, :]

    return train1, train2, test, locs


def fit_transport_map(cache_path: str | Path, locs: ArrayLike, train: ArrayLike, test: ArrayLike) -> SimpleTM:
    model_filepath = cache_path / "map" / "map.pkl"
    model_filepath.parent.mkdir(exist_ok=True, parents=True)

    if model_filepath.exists():
        logger.info(f"Loading transport map from {model_filepath}")
        with open(model_filepath, "rb") as fp:
            return pickle.load(fp)

    # Finding nearest neighbors using the `veccs` package.
    # The computation time of the model scales as a function of the condition set
    # size. We recommend restricting this to be no larger than 30 neighbors.
    largest_conditioning_set = 30
    nn = veccs.orderings.find_nns_l2(locs, largest_conditioning_set)

    tloc = torch.as_tensor(locs, dtype=torch.float32)
    tnn = torch.as_tensor(nn, dtype=torch.int64)
    nsteps = 2000

    train_data = Data.new(tloc, train, tnn)
    test_data = Data.new(tloc, test, tnn)
    tm = SimpleTM(train_data, theta_init=None, linear=False, smooth=1.5, nug_mult=4.0)
    stopper = EarlyStopper(patience=200, min_diff=3e-1)

    logger.info(f"Fitting transport map.")
    tm.fit(
        num_iter=nsteps,
        init_lr=0.01,
        test_data=test_data,
        batch_size=None,
        stopper=stopper,
    )
    
    logger.info(f"Saving transport map to {model_filepath}")
    with open(model_filepath, "wb") as fp:
        pickle.dump(tm, fp)

    return tm


def compute_yt_and_logdet(cache_path: str | Path, suffix: str, y: ArrayLike, tm: SimpleTM) -> tuple[ArrayLike, ArrayLike]:
    data_filepath = cache_path / "map" / f"yt_and_logdet-{suffix}.pkl"
    data_filepath.parent.mkdir(exist_ok=True, parents=True)

    if data_filepath.exists():
        logger.info(f"Loading data from {data_filepath}")
        with open(data_filepath, "rb") as fp:
            return pickle.load(fp)
    
    logger.info(f"Computing yt and logdet")
    with torch.no_grad():
        yt, yt_logdet = tm.compute_z_and_logdet_batched(obs=y)

    logger.info(f"Saving data to {data_filepath}")
    with open(data_filepath, "wb") as fp:
        pickle.dump((yt, yt_logdet), fp)
    
    return yt, yt_logdet


def load_map_yt_and_logdet(cache_path: str | Path, suffix: str) -> tuple[ArrayLike, ArrayLike]:
    data_filepath = cache_path / "map" / f"yt_and_logdet-{suffix}.pkl"
    data_filepath.parent.mkdir(exist_ok=True, parents=True)

    with open(data_filepath, "rb") as fp:
        return pickle.load(fp)



def fit_transformation_adaptive_switch(
    mcmc_seed: int,
    i: int,
    cache_path: str | Path,
    yt: ArrayLike,
    nparam: int = 10,
    warmup_duration: int = 1000,
    posterior_duration: int = 1000,
    thinning: int = 2,
    tau2_b_start: float = 0.2,
    tau2_b_decay_rate: float = 7.0,
    switch_threshold: float = 1.0,
) -> gs.SamplingResults:

    logger.info(f"Starting transformation fit {i=}")

    Path(cache_path).mkdir(exist_ok=True, parents=True)

    model_filepath = Path(cache_path) / "models" / f"ptm_{i}.pkl"
    results_filepath = Path(cache_path) / "results" / f"results_{i}.pkl"
    model_filepath.parent.mkdir(exist_ok=True, parents=True)
    results_filepath.parent.mkdir(exist_ok=True, parents=True)

    _, p = stats.shapiro(yt[:, i])

    if model_filepath.exists():
        logger.info(f"Loading model {i} from {model_filepath}")
        with open(model_filepath, "rb") as fp:
            ptm_i = pickle.load(fp)

        if results_filepath.exists():
            return gs.SamplingResults.pkl_load(results_filepath)

    else:
        b = 1e-6 + tau2_b_start * np.exp(-tau2_b_decay_rate * p)
        b = b.astype(np.float32)

        ptm_i = setup_ptm(
            i,
            yt[:, i],
            nparam=nparam,
            tau2_cls=VarInverseGamma,
            tau2_kwargs={"value": 1.0, "concentration": 3.0, "scale": b},
        )

        logger.info(f"Writing model {i} to {model_filepath}")
        with open(model_filepath, "wb") as fp:
            pickle.dump(ptm_i, fp)

    graph_i = ptm_i.build_graph(optimize_start_values=True)

    sample_normalization = p <= switch_threshold
    eb = gs.EngineBuilder(seed=mcmc_seed, num_chains=4)
    eb = ptm_i.setup_engine_builder(
        eb=eb, graph=graph_i, sample_normalization=sample_normalization
    )
    eb.set_duration(
        warmup_duration=warmup_duration,
        posterior_duration=posterior_duration,
        thinning_posterior=thinning,
    )

    results = cache_results(eb, filename=results_filepath, use_cache=True)

    logger.info(f"Finished transformation fit {i=}")

    return results


def compute_normalization_and_logdet(
    i: int, cache_path: str | Path, suffix: str, yt: ArrayLike | None = None
) -> tuple[ArrayLike, ArrayLike]:

    Path(cache_path).mkdir(exist_ok=True, parents=True)

    model_filepath = Path(cache_path) / "models" / f"ptm_{i}.pkl"
    results_filepath = Path(cache_path) / "results" / f"results_{i}.pkl"
    data_filepath = Path(cache_path) / f"z-{suffix}" / f"z_and_logdet_{i}.pkl"
    data_filepath.parent.mkdir(exist_ok=True, parents=True)

    if data_filepath.exists():
        logger.info(f"Returning cached transformated values {i=} from {data_filepath}")
        with open(data_filepath, "rb") as fp:
            data = pickle.load(fp)

        return data

    with open(model_filepath, "rb") as fp:
        ptm_i = pickle.load(fp)

    results = gs.SamplingResults.pkl_load(results_filepath)
    samples = results.get_posterior_samples()

    preds = PTMLocScalePredictions(samples, model=ptm_i, y=yt[:, i])

    z = preds.predict_transformation()
    z_logdet = np.log(preds.predict_transformation_deriv())

    data = (z, z_logdet)

    logger.info(f"Saving transformated values {i=} to {data_filepath}")
    with open(data_filepath, "wb") as fp:
        pickle.dump(data, fp)

    logger.info(f"Returning transformated values {i=}")
    return data


def load_normalization_and_logdet(
    cache_path: str | Path, suffix: str
) -> tuple[ArrayLike, ArrayLike]:
    """
    Returned shape:

    [location, chain sample, observation]
    """
    Path(cache_path).mkdir(exist_ok=True, parents=True)

    data_filepath = Path(cache_path) / f"z-{suffix}"

    if not data_filepath.exists() and data_filepath.is_dir():
        raise NotADirectoryError

    data_list = []

    for path in data_filepath.iterdir():
        with open(path, "rb") as fp:
            data_i = pickle.load(fp)

        data_list.append(data_i)

    z_list, z_logdet_list = zip(*data_list)

    return np.array(z_list), np.array(z_logdet_list)


def log_score_z(z: ArrayLike, logdet: ArrayLike):
    return (stats.norm.logpdf(z) + logdet)
