from tqdm import tqdm
from .legmods import SimpleTM, Data
from .btme.runners import ModelRunner
from jax import Array as JaxArray
from numpy.typing import NDArray as NumpyArray
from liesel_ptm import (
    PTMLocScale,
    PTMLocScalePredictions,
    VarInverseGamma,
    optim_flat,
    state_to_samples,
    Stopper,
    OptimResult
)
from dataclasses import dataclass, field
from scipy import stats
import numpy as np
import torch
from typing import Type

ArrayLike = JaxArray | NumpyArray


def setup_ptm(i: int, y: ArrayLike, nparam: int, tau2_cls: Type | None = None, knots: ArrayLike | None = None, **tau2_kwargs):
    if tau2_cls is None:
        tau2 = VarInverseGamma(1.0, concentration=1.0, scale=0.5, name=f"tau2_{i}")
    else:
        tau2 = tau2_cls(**tau2_kwargs, name=f"tau2_{i}")
    if knots is None:
        model: PTMLocScale = PTMLocScale.from_nparam(y=y, nparam=nparam, normalization_tau2=tau2)
        model, _ = model.optimize_knots(knot_prob_levels=(0.01, 0.99))
    else:
        model: PTMLocScale = PTMLocScale(knots, y, normalization_tau2=tau2)

    return model


@dataclass
class PTMFits:
    preds: list[PTMLocScalePredictions]
    models: list[PTMLocScale] = field(default_factory=list)
    results: list[OptimResult] = field(default_factory=list)
    fitted_params: list[dict[str, ArrayLike]] = field(default_factory=list)


class TransformationTransportMap:

    def __init__(self):
        self.batch_size: None | int = None
        self.num_epochs: None | int = None

    def set_fit(self, num_epochs: int, batch_size: int | None = None) -> None:
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def fit_map(
        self, runner: ModelRunner, train_data: Data, test_data: Data | None = None
    ) -> ArrayLike:
        assert self.num_epochs is not None, f"{self.num_epochs=} must not be None."
        test_data = test_data if test_data is None else train_data
        loss = runner.fit_model(
            train_data=train_data,
            test_data=test_data,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
        )
        return loss
    

    def fit_transformation(self, yt: ArrayLike, yt_test: ArrayLike | None = None, nparam: int = 10, max_iter: int = 10_000, patience: int = 100) -> PTMFits:
        stopper = Stopper(max_iter=max_iter, patience=patience, atol=0.5, rtol=0.01)

        models: list[PTMLocScale] = []
        preds: list[PTMLocScalePredictions] = []
        results: list[OptimResult] = []
        fitted_params: list[dict[str, ArrayLike]] = []

        for i in tqdm(range(yt.shape[1])):
            ptm_i = setup_ptm(i, yt[:, i], nparam=nparam)
            params_i = ptm_i.all_sampled_parameter_names()
            all_params_i = ptm_i.all_parameter_names()
            graph_i = ptm_i.build_graph(optimize_start_values=False)

            ptm_test_i = None
            if yt_test is not None:
                ptm_test_i = setup_ptm(i, yt_test[:, i], nparam=nparam, knots=ptm_i.knots)

            results_i = optim_flat(graph_i, params_i, stopper=stopper, model_test=ptm_test_i)
            graph_i.state = results_i.model_state
            fitted_params_i = state_to_samples(all_params_i, graph_i)
            preds_i = PTMLocScalePredictions(fitted_params_i, ptm_i)

            models.append(ptm_i)
            results.append(results_i)
            fitted_params.append(fitted_params_i)
            preds.append(preds_i)

        return PTMFits(models=models, preds=preds, results=results, fitted_params=fitted_params)

    def compute_normalization_and_logdet(
        self, fits: PTMFits, yt: ArrayLike | None = None
    ) -> tuple[ArrayLike, ArrayLike]:

        nobs = np.shape(yt)[0]
        ndim = len(fits.preds)

        z = np.zeros((ndim, nobs))
        logdet = np.zeros((ndim, nobs))

        if yt is not None:
            preds: list[PTMLocScalePredictions] = []
            for i, _ in enumerate(fits.preds):
                pred_i = PTMLocScalePredictions(
                    fits.fitted_params[i], fits.models[i], y=yt[:,i]
                )
                preds.append(pred_i)
        else:
            preds = fits.preds

        for i, pred in tqdm(enumerate(preds), total=len(preds)):
            z[i, :] = pred.predict_transformation()
            logdet[i, :] = np.log(pred.predict_transformation_deriv())

        return z.T, logdet.T

    def log_score_z(self, z: ArrayLike, logdet: ArrayLike, dim=None):
        
        return (stats.norm.logpdf(z) + logdet).sum(axis=dim)

    def fit_iteratively(self): ...

    def inverse_transformation(self, z: ArrayLike): ...

    def inverse_map(self, yt: ArrayLike): ...
