"""
Nodes for inducing points version.

- Alpha parameter: Intercept with its own hyperparameters
- Alpha parameter: Has prior mean, is estimated
- Slope has prior mean, is estimated
"""

from __future__ import annotations

import copy
from typing import Any

import jax.numpy as jnp
import liesel.model as lsl
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
from liesel_ptm.bsplines import OnionCoef, OnionKnots
from liesel_ptm.nodes import OnionCoefParam, TransformedVar, find_param

Array = Any


class Kernel(lsl.Var):
    def __init__(
        self,
        x1: lsl.Var | lsl.Node,
        x2: lsl.Var | lsl.Node,
        kernel_cls: type[tfk.AutoCompositeTensorPsdKernel],
        name: str = "",
        **kwargs,
    ) -> None:
        self.x1 = x1
        self.x2 = x2

        self.kernel_cls = kernel_cls

        def _evaluate_kernel(x1, x2, **kwargs):
            return (
                kernel_cls(**kwargs)
                .apply(x1=x1[None, :], x2=x2[:, None])
                .swapaxes(-1, -2)
            )

        calc = lsl.Calc(_evaluate_kernel, x1, x2, **kwargs).update()

        super().__init__(calc, name=name)
        self.update()


class ModelConst(lsl.Var):
    """
    A variable that holds a constant value. This value is not fitted.
    """

    def __init__(
        self,
        value: Any,
        name: str = "",
    ) -> None:
        super().__init__(value=value, name=name)
        self.parameter_names: list[str] = []
        self.hyperparameter_names: list[str] = []

    def copy_for(self, sample_locs: lsl.Var | lsl.Node) -> ModelConst:
        return ModelConst(self.value, name=self.name)  # type: ignore

    def set_locs(self, sample_locs: Array) -> ModelConst:
        return self

    def update_from(self, param: ModelConst) -> ModelVar:
        self.value = param.value  # type:ignore
        return self


class ModelVar(TransformedVar):
    """
    A variable is constant across locations.
    Can have a bijector. Does not have hyperparameters.
    """

    def __init__(
        self,
        value: Any,
        bijector: tfb.Bijector | None = tfb.Identity(),
        name: str = "",
    ) -> None:
        super().__init__(value=value, bijector=bijector, name=name)
        self.bijector = bijector
        self.parameter_names = [find_param(self).name]
        self.hyperparameter_names: list[str] = []

    def copy_for(self, sample_locs: lsl.Var | lsl.Node) -> ModelConst:
        val = self.value  # type: ignore
        bij = self.bijector
        return ModelVar(val, bijector=bij, name=self.name)

    def set_locs(self, sample_locs: Array) -> ModelConst:
        return self

    def update_from(self, param: ModelVar) -> ModelVar:
        if self.strong:
            self.value = param.value  # type: ignore
            return self
        if self.transformed:
            self.transformed.value = param.transformed.value
            return self.update()

        raise RuntimeError("Cannot update; variable is neither strong nor transformed.")


class ParamPredictivePointProcessGP(lsl.Var):
    def __init__(
        self,
        inducing_locs: lsl.Var | lsl.Node,
        sample_locs: lsl.Var | lsl.Node,
        kernel_cls: type[tfk.AutoCompositeTensorPsdKernel],
        bijector: tfb.Bijector = tfb.Identity(),
        name: str = "",
        expand_dims: bool = True,
        **kernel_params: lsl.Var | TransformedVar,
    ) -> None:
        kernel_uu = Kernel(
            x1=inducing_locs,
            x2=inducing_locs,
            kernel_cls=kernel_cls,
            **kernel_params,
            name=f"{name}_kernel",
        ).update()

        kernel_du = Kernel(
            x1=sample_locs,
            x2=inducing_locs,
            kernel_cls=kernel_cls,
            **kernel_params,
            name=f"{name}_kernel_latent_u",
        ).update()

        n_inducing_locs = kernel_uu.value.shape[0]

        self.latent_var = lsl.param(
            jnp.zeros((n_inducing_locs,)),
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name=f"{name}_latent",
        )

        # a small value added to the diagonal of Kuu for numerical stability
        salt = jnp.diag(jnp.full(shape=(n_inducing_locs,), fill_value=1e-6))

        self.mean = lsl.param(0.0, name=f"{name}_mean")

        def _compute_param(latent_var, Kuu, Kdu, mean):
            Kuu = Kuu + salt
            L = jnp.linalg.cholesky(Kuu)
            Li = jnp.linalg.inv(L)

            value = bijector.forward(Kdu @ Li.T @ latent_var)

            if expand_dims:
                value = jnp.expand_dims(value, -1)

            return value + mean

        super().__init__(
            lsl.Calc(_compute_param, self.latent_var, kernel_uu, kernel_du, self.mean),
            name=name,
        )

        self.bijector = bijector
        self.kernel_params = kernel_params
        self.kernel_cls = kernel_cls
        self.inducing_locs = inducing_locs
        self.sample_locs = sample_locs
        self.K = n_inducing_locs
        self.parameter_names = [self.latent_var.name, self.mean.name]

    @property
    def hyperparameter_names(self):
        return [find_param(param).name for param in self.kernel_params.values()]

    def copy_for(
        self, sample_locs: lsl.Var | lsl.Node | None = None
    ) -> ParamPredictivePointProcessGP:
        kernel_params = {
            name: copy.deepcopy(param) for name, param in self.kernel_params.items()
        }

        sample_locs = (
            sample_locs if sample_locs is not None else copy.deepcopy(self.sample_locs)
        )

        var = ParamPredictivePointProcessGP(
            inducing_locs=copy.deepcopy(self.inducing_locs),
            sample_locs=sample_locs,
            kernel_cls=self.kernel_cls,
            bijector=self.bijector,
            name=self.name,
            **kernel_params,
        )

        var.latent_var.value = self.latent_var.value

        return var

    def set_locs(self, sample_locs: Array) -> ModelConst:
        self.sample_locs.value = sample_locs
        return self.update()

    def update_from(
        self, param: ParamPredictivePointProcessGP
    ) -> ParamPredictivePointProcessGP:
        for name, kernel_param in param.kernel_params.items():
            self.kernel_params[name].value = kernel_param.value
            self.kernel_params[name].update()

        self.latent_var.value = param.latent_var.value
        self.update()

        return self


def brownian_motion_mat(nrows: int, ncols: int):
    r = jnp.arange(nrows)[:, None] + 1
    c = jnp.arange(ncols)[None, :] + 1
    return jnp.minimum(r, c)


def rw_weight_matrix(D: int):
    C = jnp.eye(D - 1) - jnp.ones(D - 1) / (D - 1)
    B = brownian_motion_mat(D - 2, D - 2)
    L = jnp.linalg.cholesky(B, upper=False)
    W = C @ jnp.r_[jnp.zeros((1, D - 2)), L]
    return W


class RandomWalkParamPredictivePointProcessGP(lsl.Var):
    """
    Assumes the intrinsic model of coregionalization.

    Params
    ------
    D
        Dimension of the random walk.
    K
        Number of inducing locations.
    """

    def __init__(
        self,
        inducing_locs: lsl.Var | lsl.Node,
        sample_locs: lsl.Var | lsl.Node,
        D: int,
        kernel_cls: type[tfk.AutoCompositeTensorPsdKernel],
        name: str = "",
        **kernel_params: lsl.Var | TransformedVar,
    ):
        kernel_uu = Kernel(
            x1=inducing_locs,
            x2=inducing_locs,
            kernel_cls=kernel_cls,
            **kernel_params,
            name=f"{name}_kernel",
        )

        kernel_du = Kernel(
            x1=sample_locs,
            x2=inducing_locs,
            kernel_cls=kernel_cls,
            **kernel_params,
            name=f"{name}_kernel_latent_u",
        )

        W = rw_weight_matrix(D)

        nrow_W = W.shape[0]

        n_inducing_locs = kernel_uu.value.shape[0]
        n_sample_locs = kernel_du.value.shape[0]

        latent_var = lsl.param(
            jnp.zeros((n_inducing_locs * (W.shape[1]),)),
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name=f"{name}_latent",
        )

        salt = jnp.eye(kernel_uu.value.shape[0]) * 1e-6

        def _compute_param(latent_var, Kuu, Kdu):
            Kuu = Kuu + salt
            L = jnp.linalg.cholesky(Kuu)
            Li = jnp.linalg.inv(L)

            Kdu_uui = jnp.kron(W, (Kdu @ Li.T))
            delta_long = Kdu_uui @ latent_var
            delta_mat = jnp.reshape(delta_long, (nrow_W, n_sample_locs))

            return delta_mat

        super().__init__(
            lsl.Calc(
                _compute_param,
                latent_var=latent_var,
                Kuu=kernel_uu,
                Kdu=kernel_du,
            ),
            name=name,
        )
        self.update()

        self.latent_var = latent_var
        self.kernel_uu = kernel_uu
        self.kernel_du = kernel_du
        self.W = W
        self.kernel_params = kernel_params
        self.kernel_cls = kernel_cls
        self.inducing_locs = inducing_locs
        self.sample_locs = sample_locs
        self.K = n_inducing_locs

        self.parameter_names = [latent_var.name]

    @property
    def hyperparameter_names(self):
        return [find_param(param).name for param in self.kernel_params.values()]


class OnionCoefPredictivePointProcessGP(lsl.Var):
    def __init__(
        self,
        latent_coef: RandomWalkParamPredictivePointProcessGP,
        coef_spec: OnionCoef,
        name: str = "",
    ) -> None:
        super().__init__(
            lsl.Calc(lambda latent: coef_spec(latent.T), latent_coef).update(),
            name=name,
        )

        self.coef_spec = coef_spec
        self.latent_coef = latent_coef
        self.parameter_names = latent_coef.parameter_names

    @property
    def hyperparameter_names(self):
        return self.latent_coef.hyperparameter_names

    @classmethod
    def new_from_locs(
        cls,
        knots: OnionKnots,
        inducing_locs: lsl.Var | lsl.Node,
        sample_locs: lsl.Var | lsl.Node,
        kernel_cls: type[tfk.AutoCompositeTensorPsdKernel],
        name: str = "",
        **kernel_params: lsl.Var | TransformedVar,
    ) -> OnionCoefPredictivePointProcessGP:
        coef_spec = OnionCoef(knots)

        latent_coef = RandomWalkParamPredictivePointProcessGP(
            inducing_locs=inducing_locs,
            sample_locs=sample_locs,
            D=knots.nparam + 1,
            kernel_cls=kernel_cls,
            name=f"{name}_log_increments",
            **kernel_params,
        )

        return cls(coef_spec=coef_spec, latent_coef=latent_coef, name=name)

    def copy_for(
        self, sample_locs: lsl.Var | lsl.Node | None = None
    ) -> OnionCoefPredictivePointProcessGP:
        coef_spec = OnionCoef(self.coef_spec.knots)

        kernel_params = {
            name: copy.deepcopy(param)
            for name, param in self.latent_coef.kernel_params.items()
        }

        sample_locs = (
            sample_locs
            if sample_locs is not None
            else copy.deepcopy(self.latent_coef.sample_locs)
        )

        latent_coef = RandomWalkParamPredictivePointProcessGP(
            inducing_locs=copy.deepcopy(self.latent_coef.inducing_locs),
            sample_locs=sample_locs,
            D=self.coef_spec.knots.nparam + 1,
            kernel_cls=self.latent_coef.kernel_cls,
            name=f"{self.name}_log_increments",
            **kernel_params,
        )

        latent_coef.latent_var.value = self.latent_coef.latent_var.value

        return OnionCoefPredictivePointProcessGP(
            coef_spec=coef_spec, latent_coef=latent_coef, name=self.name
        )

    def update_from(
        self, coef: OnionCoefPredictivePointProcessGP
    ) -> OnionCoefPredictivePointProcessGP:
        for name, kernel_param in coef.latent_coef.kernel_params.items():
            self.latent_coef.kernel_params[name].value = kernel_param.value
            self.latent_coef.kernel_params[name].update()

        self.latent_coef.latent_var.value = coef.latent_coef.latent_var.value
        self.latent_coef.update()
        self.update()

        return self

    def spawn_intercept(
        self, name: str = "intercept", **kernel_params
    ) -> ParamPredictivePointProcessGP:
        if not kernel_params:
            kernel_params = self.latent_coef.kernel_params

        intercept = ParamPredictivePointProcessGP(
            inducing_locs=self.latent_coef.inducing_locs,
            sample_locs=self.latent_coef.sample_locs,
            kernel_cls=self.latent_coef.kernel_cls,
            name=name,
            **kernel_params,
        )

        return intercept

    def spawn_slope(
        self,
        bijector: tfb.Bijector = tfb.Softplus(),
        name: str = "slope",
        **kernel_params,
    ) -> ParamPredictivePointProcessGP:
        if not kernel_params:
            kernel_params = self.latent_coef.kernel_params

        slope = ParamPredictivePointProcessGP(
            inducing_locs=self.latent_coef.inducing_locs,
            sample_locs=self.latent_coef.sample_locs,
            kernel_cls=self.latent_coef.kernel_cls,
            name=name,
            bijector=bijector,
            **kernel_params,
        )

        return slope


class ModelOnionCoef(OnionCoefParam):
    def __init__(self, knots: OnionKnots, name: str = "") -> None:
        super().__init__(knots=knots, tau2=lsl.Var(1.0, name="tau2_onion"), name=name)

        self.parameter_names: list[str] = [self.log_increments.transformed.name]
        self.hyperparameter_names: list[str] = []
