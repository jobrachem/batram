from pathlib import Path
import numpy as np
import liesel_ptm as ptm
import jax
import liesel.goose as gs
import jax.numpy as jnp
from liesel.goose.types import KeyArray, Array
import pandas as pd
import tensorflow_probability.substrates.jax.distributions as tfd
import click
import logging
from liesel.logging import add_file_handler


normal = tfd.Normal(loc=0.0, scale=1.0)

def draw_sample(
    key: KeyArray,
    shape: Array,
    nobs: int,
) -> pd.DataFrame:

    dg = ptm.PTMLocScaleDataGen(shape, loc_fn=lambda x: 3 * np.sin(x), ncov=1)
    sample = dg.sample(key=key, nobs=nobs)
    df = dg.to_df(sample)

    return df


def train_test_split(
    df: pd.DataFrame, test_share: float = 0.3
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = df.shape[0]

    ntest = int(test_share * n)

    ntrain = n - ntest
    ntrain1 = ntrain // 2
    ntrain2 = ntrain1

    while (ntrain1 + ntrain2 + ntest) < n:
        ntest += 1

    df_train1 = df.iloc[:ntrain1, :]
    df_train2 = df.iloc[ntrain1 : (ntrain1 + ntrain2), :]
    df_test = df.iloc[-ntest:, :]

    assert df_train1.shape[0] == ntrain1
    assert df_train2.shape[0] == ntrain2
    assert df_test.shape[0] == ntest

    return df_train1, df_train2, df_test


def setup_loc_model(df: pd.DataFrame) -> ptm.PTMLocScale:
    model = ptm.PTMLocScale.from_nparam(
        y=df.y.to_numpy(),
        nparam=15,
        normalization_tau2=ptm.VarInverseGamma(
            value=1.0, concentration=3.0, scale=0.2, name="tau2"
        ),
        scaling_factor=ptm.TruncatedNormalOmega(name="omega")
    )

    model.loc_model += ptm.PSpline(
        x=df.x0.to_numpy(),
        nparam=20,
        tau2=ptm.VarInverseGamma(
            value=1.0, concentration=1.0, scale=0.5, name="tau2_x0"
        ),
        name="x0",
    )
    return model


def fit_loc_model(model: ptm.PTMLocScale) -> dict[str, Array]:

    graph = model.build_graph(optimize_start_values=False)

    stopper = ptm.Stopper(max_iter=10_000, patience=500, atol=0.001)
    results = ptm.optim_flat(
        model=graph,
        params=[
            "x0_coef",
            "tau2_x0_transformed",
        ],
        stopper=stopper,
    )
    graph.state = results.model_state

    samples = ptm.state_to_samples(model.all_parameter_names(), graph)
    return samples


def residuals_loc_model(
    df: pd.DataFrame, model: ptm.PTMLocScale, samples: dict[str, Array]
) -> Array:
    pred = ptm.PTMLocScalePredictions(
        samples, model, y=df.y.to_numpy(), x0=df.x0.to_numpy()
    )
    return pred.predict_residuals().squeeze()


def setup_dist_model(
    residuals: Array, tau2_a: float = 3.0, tau2_b: float = 0.2
) -> ptm.PTMLocScale:
    model = ptm.PTMLocScale.from_nparam(
        y=residuals,
        nparam=15,
        normalization_tau2=ptm.VarInverseGamma(
            value=1.0, concentration=tau2_a, scale=tau2_b, name="tau2"
        ),
        scaling_factor=ptm.TruncatedNormalOmega(name="omega")
    )

    model, _ = model.optimize_knots(optimize_params=[])
    return model


def fit_dist_model(model: ptm.PTMLocScale) -> dict[str, Array]:
    graph = model.build_graph(optimize_start_values=False)

    stopper = ptm.Stopper(max_iter=10_000, patience=500, atol=0.001)
    result = ptm.optim_flat(
        model=graph,
        params=["normalization_shape_transformed", "tau2_transformed", "omega_transformed"],
        stopper=stopper,
    )

    graph.state = result.model_state

    samples = ptm.state_to_samples(model.all_parameter_names(), graph)
    return samples


def fit_combined_model(model: ptm.PTMLocScale) -> dict[str, Array]:
    graph = model.build_graph(optimize_start_values=False)

    stopper = ptm.Stopper(max_iter=10_000, patience=500, atol=0.001)
    result = ptm.optim_flat(
        model=graph,
        params=[
            "x0_coef",
            "tau2_x0_transformed",
            "normalization_shape_transformed",
            "tau2_transformed",
            "omega_transformed"
        ],
        stopper=stopper,
    )

    graph.state = result.model_state

    samples = ptm.state_to_samples(model.all_parameter_names(), graph)
    return samples


def score_loc_model(
    test_df: pd.DataFrame, model: ptm.PTMLocScale, samples: dict[str, Array]
) -> Array:
    pred = ptm.PTMLocScalePredictions(
        samples, model, y=test_df.y.to_numpy(), x0=test_df.x0.to_numpy()
    )
    return -pred.predict_log_prob().mean()


def score_loc_model_manual(
    test_df: pd.DataFrame, model: ptm.PTMLocScale, samples: dict[str, Array]
) -> Array:
    pred = ptm.PTMLocScalePredictions(
        samples, model, y=test_df.y.to_numpy(), x0=test_df.x0.to_numpy()
    )

    z = pred.predict_transformation()
    z_deriv = pred.predict_transformation_deriv()

    log_prob = model.refdist.log_prob(z) + np.log(z_deriv)
    return -log_prob.mean()

def z_and_deriv(
        df: pd.DataFrame, model: ptm.PTMLocScale, samples: dict[str, Array]
) -> tuple[Array, Array]:
    
    pred = ptm.PTMLocScalePredictions(
        samples, model, y=df.y.to_numpy(), x0=df.x0.to_numpy()
    )

    z = pred.predict_transformation()
    z_deriv = pred.predict_transformation_deriv()
    return z, z_deriv

    

def identity_decorator(func):
    # This decorator simply returns the original function
    return func


def score_combined_model(
    test_df: pd.DataFrame,
    loc_model: ptm.PTMLocScale,
    loc_samples: dict[str, Array],
    dist_samples: dict[str, Array],
) -> Array:

    samples = loc_samples.copy()

    samples["tau2_transformed"] = dist_samples["tau2_transformed"]
    samples["normalization_shape_transformed"] = dist_samples[
        "normalization_shape_transformed"
    ]
    samples["unscaled_normalization_mean"] = dist_samples["unscaled_normalization_mean"]
    samples["unscaled_normalization_sd"] = dist_samples["unscaled_normalization_sd"]

    pred = ptm.PTMLocScalePredictions(
        samples, loc_model, y=test_df.y.to_numpy(), x0=test_df.x0.to_numpy()
    )
    return -pred.predict_log_prob().mean()


def sample_shape_array(key: KeyArray, nshape: int, scale: float) -> Array:
    return ptm.sample_shape(key, nshape=nshape, scale=scale).sample

def compute_log_score(z, z_deriv):
    return -(normal.log_prob(z) + np.log(z_deriv)).mean()

def compute_dist_model_log_score(eps, eps_deriv, dist_model, dist_samples):
    pred = ptm.PTMLocScalePredictions(
        dist_samples, dist_model, y=eps
    )

    z = pred.predict_transformation()
    z_deriv = pred.predict_transformation_deriv()

    return -(normal.log_prob(z) + np.log(z_deriv) + np.log(eps_deriv)).mean()



def run_one_simulation(
    data_seed: int,
    shape_seed: int,
    nobs: int,
    dist_model_tau2_a: float = 3.0,
    dist_model_tau2_b: float = 0.2,
    cache_path: str | Path | None = None,
    data_cache_path: str | Path | None = None,
    test_share: float = 0.3,
) -> dict[str, float | int]:
    if cache_path is not None:
        cache = ptm.cache(cache_path)
    else:
        cache = identity_decorator

    if data_cache_path is not None:
        data_cache = ptm.cache(data_cache_path)
    else:
        data_cache = cache

    shape_key = jax.random.PRNGKey(shape_seed)
    shape = data_cache(sample_shape_array)(shape_key, nshape=10, scale=0.5)

    data_key = jax.random.PRNGKey(data_seed)
    df = data_cache(draw_sample)(key=data_key, shape=shape, nobs=nobs)

    train1, train2, test = train_test_split(df, test_share=test_share)

    loc_model = setup_loc_model(train1)
    loc_samples = fit_loc_model(loc_model)

    eps_test, eps_test_deriv = z_and_deriv(test, model=loc_model, samples=loc_samples)
    eps_train1, eps_train1_deriv = z_and_deriv(train1, model=loc_model, samples=loc_samples)
    eps_train2, eps_train2_deriv = z_and_deriv(train2, model=loc_model, samples=loc_samples)

    loc_score_test = compute_log_score(eps_test, eps_test_deriv)
    loc_score_train1 = compute_log_score(eps_train1, eps_train1_deriv)
    loc_score_train2 = compute_log_score(eps_train2, eps_train2_deriv)

    combined_samples = fit_combined_model(loc_model)
    combined_score = score_loc_model(test_df=test, model=loc_model, samples=combined_samples)

    dist_model_train1 = setup_dist_model(
        eps_train1, tau2_a=dist_model_tau2_a, tau2_b=dist_model_tau2_b
    )
    dist_samples_train1 = fit_dist_model(dist_model_train1)

    
    dist_model_train2 = setup_dist_model(
        eps_train2, tau2_a=dist_model_tau2_a, tau2_b=dist_model_tau2_b
    )
    dist_samples_train2 = fit_dist_model(dist_model_train2)

    dist_score_test_trained_on_train1 = compute_dist_model_log_score(eps_test, eps_test_deriv, dist_model_train1, dist_samples_train1)
    dist_score_test_trained_on_train2 = compute_dist_model_log_score(eps_test, eps_test_deriv, dist_model_train2, dist_samples_train2)

    dist_score_train1_trained_on_train1 = compute_dist_model_log_score(eps_train1, eps_train1_deriv, dist_model_train1, dist_samples_train1)
    dist_score_train2_trained_on_train2 = compute_dist_model_log_score(eps_train2, eps_train2_deriv, dist_model_train2, dist_samples_train2)

    data = {}
    data["loc_score_test"] = float(loc_score_test)
    data["loc_score_train1"] = float(loc_score_train1)
    data["loc_score_train2"] = float(loc_score_train2)
    
    data["combined_score_test"] = float(combined_score)
    data["dist_score_test_trained_on_train1"] = float(dist_score_test_trained_on_train1)
    data["dist_score_test_trained_on_train2"] = float(dist_score_test_trained_on_train2)

    data["dist_score_train1_trained_on_train1"] = float(dist_score_train1_trained_on_train1)
    data["dist_score_train2_trained_on_train2"] = float(dist_score_train2_trained_on_train2)

    data["shape_seed"] = shape_seed
    data["data_seed"] = data_seed
    data["nobs"] = nobs

    data["test_share"] = test_share
    data["n_train1"] = train1.shape[0]
    data["n_train2"] = train2.shape[0]
    data["n_test"] = test.shape[0]

    return data


def setup_logging(data_seed: int, shape_seed: int, path: Path | str):
    logger = logging.getLogger("ttm_sim")
    logger.setLevel(logging.INFO)

    path = Path(path).resolve()
    logs = path
    logs.mkdir(exist_ok=True, parents=False)

    try:
        simlogger_stdout = logger.handlers[0]
    except IndexError:
        simlogger_stdout = None

    logfile = logs / f"log-data_seed_{data_seed}-shape_seed_{shape_seed}.log"
    add_file_handler(path=logfile, level="info", logger="ttm_sim")

    if simlogger_stdout is not None:
        logger.removeHandler(simlogger_stdout)


@click.command()
@click.option(
    "--data_seed", help="Seed for random number generation.", required=True, type=int
)
@click.option(
    "--shape_seed", help="Seed for random number generation.", required=True, type=int
)
@click.option("--nobs", help="Sample size to use.", required=True, type=int)
@click.option(
    "--path", help="Directory.", required=True, type=str
)
def run(data_seed, shape_seed, nobs, path):
    path = Path(path)
    out_path = path / "out" / f"results-data{data_seed}-shape{shape_seed}-nobs{nobs}.csv"
    
    logger = logging.getLogger("ttm_sim")
    setup_logging(data_seed=data_seed, shape_seed=shape_seed, path=path / "logs")

    if out_path.exists():
        logger.info(f"SKIPPING, because {out_path=} exists.")
        return

    logger.info(f"STARTING {data_seed=}, {shape_seed=}")
    data = run_one_simulation(data_seed=data_seed, shape_seed=shape_seed, nobs=nobs)


    out_path.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(data, index=[0]).to_csv(out_path, index=False)

    logger.info(f"FINISHED {data_seed=}, {shape_seed=}")
    
    
if __name__ == "__main__":
    run()