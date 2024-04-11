from dataclasses import dataclass, field

import numpy as np
from jax import Array as JaxArray
from joblib import Parallel, delayed
from liesel_ptm import (
    OptimResult,
    PTMLocScale,
    PTMLocScalePredictions,
    Stopper,
    VarInverseGamma,
    optim_flat,
    state_to_samples,
)
from numpy.typing import NDArray as NumpyArray
from scipy import stats
from tqdm import tqdm

from .btme.runners import ModelRunner
from .legmods import Data

ArrayLike = JaxArray | NumpyArray


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


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


@dataclass
class PTMFits:
    preds: list[PTMLocScalePredictions]
    models: list[PTMLocScale] = field(default_factory=list)
    results: list[OptimResult | None] = field(default_factory=list)
    fitted_params: list[dict[str, ArrayLike]] = field(default_factory=list)


@dataclass
class OnePTMFit:
    pred: PTMLocScalePredictions
    model: PTMLocScale | None = None
    result: OptimResult | None = None
    fitted_params: dict[str, ArrayLike] | None = None


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
            train_data=train_data,  # type: ignore
            test_data=test_data,  # type: ignore
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
        )
        return loss

    def fit_transformation(
        self,
        yt: ArrayLike,
        yt_test: ArrayLike | None = None,
        nparam: int = 10,
        max_iter: int = 10_000,
        patience: int = 100,
        tau2_a: float = 3.0,
        tau2_b: float = 0.2,
    ) -> PTMFits:
        stopper = Stopper(max_iter=max_iter, patience=patience, atol=0.001, rtol=0.001)

        models: list[PTMLocScale] = []
        preds: list[PTMLocScalePredictions] = []
        results: list[OptimResult] = []
        fitted_params: list[dict[str, ArrayLike]] = []

        for i in tqdm(range(yt.shape[1])):
            ptm_i = setup_ptm(
                i,
                yt[:, i],
                nparam=nparam,
                tau2_cls=VarInverseGamma,
                tau2_kwargs={"value": 1.0, "concentration": tau2_a, "scale": tau2_b},
            )
            params_i = ptm_i.all_sampled_parameter_names()
            all_params_i = ptm_i.all_parameter_names()
            graph_i = ptm_i.build_graph(optimize_start_values=False)

            ptm_test_i = None
            if yt_test is not None:
                ptm_test_i = setup_ptm(
                    i, yt_test[:, i], nparam=nparam, knots=ptm_i.knots
                )

            results_i = optim_flat(
                graph_i, params_i, stopper=stopper, model_test=ptm_test_i
            )
            graph_i.state = results_i.model_state
            fitted_params_i = state_to_samples(all_params_i, graph_i)
            preds_i = PTMLocScalePredictions(fitted_params_i, ptm_i)

            models.append(ptm_i)
            results.append(results_i)
            fitted_params.append(fitted_params_i)
            preds.append(preds_i)

        return PTMFits(
            models=models, preds=preds, results=results, fitted_params=fitted_params
        )

    def fit_transformation_switch(
        self,
        yt: ArrayLike,
        yt_test: ArrayLike | None = None,
        nparam: int = 10,
        max_iter: int = 10_000,
        patience: int = 1000,
    ) -> PTMFits:
        stopper = Stopper(max_iter=max_iter, patience=patience, atol=0.001, rtol=0.001)

        models: list[PTMLocScale] = []
        preds: list[PTMLocScalePredictions] = []
        results: list[OptimResult | None] = []
        fitted_params: list[dict[str, ArrayLike]] = []

        for i in tqdm(range(yt.shape[1])):
            ptm_i = setup_ptm(
                i,
                yt[:, i],
                nparam=nparam,
                tau2_cls=VarInverseGamma,
                tau2_kwargs={"value": 1.0, "concentration": 3.0, "scale": 0.2},
            )

            params_i = ptm_i.all_sampled_parameter_names()
            all_params_i = ptm_i.all_parameter_names()
            graph_i = ptm_i.build_graph(optimize_start_values=False)

            ptm_test_i = None
            if yt_test is not None:
                ptm_test_i = setup_ptm(
                    i, yt_test[:, i], nparam=nparam, knots=ptm_i.knots
                )

            _, p = stats.shapiro(yt[:, i])

            results_i = None

            if p < 0.05:
                results_i = optim_flat(
                    graph_i, params_i, stopper=stopper, model_test=ptm_test_i
                )
                graph_i.state = results_i.model_state

            fitted_params_i = state_to_samples(all_params_i, graph_i)
            preds_i = PTMLocScalePredictions(fitted_params_i, ptm_i)

            models.append(ptm_i)
            results.append(results_i)
            fitted_params.append(fitted_params_i)
            preds.append(preds_i)

        return PTMFits(
            models=models, preds=preds, results=results, fitted_params=fitted_params
        )

    def fit_transformation_adaptive_switch(
        self,
        yt: ArrayLike,
        yt_test: ArrayLike | None = None,
        nparam: int = 10,
        max_iter: int = 10_000,
        patience: int = 1000,
        tau2_b_start: float = 0.2,
        tau2_b_decay_rate: float = 7.0,
        switch_threshold: float = 0.05,
    ) -> PTMFits:
        stopper = Stopper(max_iter=max_iter, patience=patience, atol=0.001, rtol=0.001)

        models: list[PTMLocScale] = []
        preds: list[PTMLocScalePredictions] = []
        results: list[OptimResult | None] = []
        fitted_params: list[dict[str, ArrayLike]] = []

        for i in tqdm(range(yt.shape[1])):
            _, p = stats.shapiro(yt[:, i])
            b = 1e-6 + tau2_b_start * np.exp(-tau2_b_decay_rate * p)
            b = b.astype(np.float32)

            ptm_i = setup_ptm(
                i,
                yt[:, i],
                nparam=nparam,
                tau2_cls=VarInverseGamma,
                tau2_kwargs={"value": 1.0, "concentration": 3.0, "scale": b},
            )

            params_i = ptm_i.all_sampled_parameter_names()
            all_params_i = ptm_i.all_parameter_names()
            graph_i = ptm_i.build_graph(optimize_start_values=False)

            ptm_test_i = None
            if yt_test is not None:
                ptm_test_i = setup_ptm(
                    i, yt_test[:, i], nparam=nparam, knots=ptm_i.knots
                )

            _, p = stats.shapiro(yt[:, i])

            results_i = None

            if p < switch_threshold:
                results_i = optim_flat(
                    graph_i, params_i, stopper=stopper, model_test=ptm_test_i
                )
                graph_i.state = results_i.model_state

            fitted_params_i = state_to_samples(all_params_i, graph_i)
            preds_i = PTMLocScalePredictions(fitted_params_i, ptm_i)

            models.append(ptm_i)
            results.append(results_i)
            fitted_params.append(fitted_params_i)
            preds.append(preds_i)

        return PTMFits(
            models=models, preds=preds, results=results, fitted_params=fitted_params
        )

    def fit_transformation_parallel(
        self,
        n_jobs: int,
        yt: ArrayLike,
        yt_test: ArrayLike | None = None,
        nparam: int = 10,
        max_iter: int = 10_000,
        patience: int = 1000,
        tau2_b_start: float = 0.2,
        tau2_b_decay_rate: float = 7.0,
        switch_threshold: float = 0.05,
    ):
        stopper = Stopper(max_iter=max_iter, patience=patience, atol=0.001, rtol=0.001)

        @delayed
        def fn(i: int):
            _, p = stats.shapiro(yt[:, i])
            b = 1e-6 + tau2_b_start * np.exp(-tau2_b_decay_rate * p)
            b = b.astype(np.float32)

            ptm_i = setup_ptm(
                i,
                yt[:, i],
                nparam=nparam,
                tau2_cls=VarInverseGamma,
                tau2_kwargs={"value": 1.0, "concentration": 3.0, "scale": b},
            )

            params_i = ptm_i.all_sampled_parameter_names()
            all_params_i = ptm_i.all_parameter_names()
            graph_i = ptm_i.build_graph(optimize_start_values=False)

            ptm_test_i = None
            if yt_test is not None:
                ptm_test_i = setup_ptm(
                    i, yt_test[:, i], nparam=nparam, knots=ptm_i.knots
                )

            _, p = stats.shapiro(yt[:, i])

            results_i = None

            if p < switch_threshold:
                results_i = optim_flat(
                    graph_i, params_i, stopper=stopper, model_test=ptm_test_i
                )
                graph_i.state = results_i.model_state

            fitted_params_i = state_to_samples(all_params_i, graph_i)
            preds_i = PTMLocScalePredictions(fitted_params_i, ptm_i)

            return OnePTMFit(
                pred=preds_i,
                model=ptm_i,
                result=results_i,
                fitted_params=fitted_params_i,
            )

        parallel = ProgressParallel(n_jobs=n_jobs)

        generator = (fn(i) for i in range(yt.shape[1]))

        return parallel(generator)

    def fit_transformation_adaptive(
        self,
        yt: ArrayLike,
        yt_test: ArrayLike | None = None,
        nparam: int = 10,
        max_iter: int = 10_000,
        patience: int = 100,
        tau2_b_start: float = 0.3,
        tau2_b_decay_rate: float = 7.0,
    ) -> PTMFits:
        stopper = Stopper(max_iter=max_iter, patience=patience, atol=0.001, rtol=0.001)

        models: list[PTMLocScale] = []
        preds: list[PTMLocScalePredictions] = []
        results: list[OptimResult | None] = []
        fitted_params: list[dict[str, ArrayLike]] = []

        for i in tqdm(range(yt.shape[1])):
            _, p = stats.shapiro(yt[:, i])
            b = 1e-6 + tau2_b_start * np.exp(-tau2_b_decay_rate * p)
            b = b.astype(np.float32)

            ptm_i = setup_ptm(
                i,
                yt[:, i],
                nparam=nparam,
                tau2_cls=VarInverseGamma,
                tau2_kwargs={"value": 1.0, "concentration": 3.0, "scale": b},
            )

            params_i = ptm_i.all_sampled_parameter_names()
            all_params_i = ptm_i.all_parameter_names()
            graph_i = ptm_i.build_graph(optimize_start_values=False)

            ptm_test_i = None
            if yt_test is not None:
                ptm_test_i = setup_ptm(
                    i, yt_test[:, i], nparam=nparam, knots=ptm_i.knots
                )

            results_i = optim_flat(
                graph_i, params_i, stopper=stopper, model_test=ptm_test_i
            )
            graph_i.state = results_i.model_state

            fitted_params_i = state_to_samples(all_params_i, graph_i)
            preds_i = PTMLocScalePredictions(fitted_params_i, ptm_i)

            models.append(ptm_i)
            results.append(results_i)
            fitted_params.append(fitted_params_i)
            preds.append(preds_i)

        return PTMFits(
            models=models, preds=preds, results=results, fitted_params=fitted_params
        )

    def compute_normalization_and_logdet(
        self, fits: PTMFits, yt: ArrayLike | None = None
    ) -> tuple[ArrayLike, ArrayLike]:
        nobs = np.shape(yt)[0]  # type: ignore
        ndim = len(fits.preds)

        z = np.zeros((ndim, nobs))
        logdet = np.zeros((ndim, nobs))

        if yt is not None:
            preds: list[PTMLocScalePredictions] = []
            for i, _ in enumerate(fits.preds):
                pred_i = PTMLocScalePredictions(
                    fits.fitted_params[i], fits.models[i], y=yt[:, i]
                )
                preds.append(pred_i)
        else:
            preds = fits.preds

        for i, pred in tqdm(enumerate(preds), total=len(preds)):
            z[i, :] = pred.predict_transformation()
            logdet[i, :] = np.log(pred.predict_transformation_deriv())

        return z.T, logdet.T

    def compute_normalization_and_logdet_parallel(
        self, n_jobs: int, fits: list[OnePTMFit], yt: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        ndim = len(fits)

        @delayed
        def fn(i: int):
            pred = PTMLocScalePredictions(
                fits[i].fitted_params, fits[i].model, y=yt[:, i]
            )
            z_i = pred.predict_transformation()
            logdet_i = np.log(pred.predict_transformation_deriv())
            return z_i.squeeze(), logdet_i.squeeze()

        parallel = ProgressParallel(n_jobs=n_jobs)
        generator = (fn(i) for i in range(ndim))

        results = parallel(generator)
        z_list, logdet_list = zip(*results)

        return np.array(z_list).T, np.array(logdet_list).T

    def compute_normalization_and_logdet_nonparallel(
        self, n_jobs: int, fits: list[OnePTMFit], yt: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        ndim = len(fits)

        def fn(i: int):
            pred = PTMLocScalePredictions(
                fits[i].fitted_params, fits[i].model, y=yt[:, i]
            )
            z_i = pred.predict_transformation()
            logdet_i = np.log(pred.predict_transformation_deriv())
            return z_i.squeeze(), logdet_i.squeeze()

        results = [fn(i) for i in tqdm(range(ndim), total=ndim)]

        z_list, logdet_list = zip(*results)

        return np.array(z_list).T, np.array(logdet_list).T

    def log_score_z(self, z: ArrayLike, logdet: ArrayLike, dim=None):
        return (stats.norm.logpdf(z) + logdet).sum(axis=dim)

    def fit_iteratively(self):
        ...

    def inverse_transformation(self, z: ArrayLike):
        ...

    def inverse_map(self, yt: ArrayLike):
        ...
