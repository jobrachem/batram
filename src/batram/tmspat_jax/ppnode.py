"""
Nodes for inducing points version.

- Alpha parameter: Intercept with its own hyperparameters
- Alpha parameter: Has prior mean, is estimated
- Slope has prior mean, is estimated
"""

from __future__ import annotations

from typing import Any

import liesel.model as lsl
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
import jax.numpy as jnp
from liesel_ptm.nodes import TransformedVar, find_param
from liesel_ptm.bsplines import OnionCoef, OnionKnots

Array = Any


class Kernel(lsl.Var):
    def __init__(
        self,
        x1: Array,
        x2: Array,
        kernel_cls: type[tfk.AutoCompositeTensorPsdKernel],
        name: str = "",
        **kwargs,
    ) -> None:

        self.x1 = x1
        self.x2 = x2

        self.kernel_cls = kernel_cls

        def _evaluate_kernel(**kwargs):
            return (
                kernel_cls(**kwargs)
                .apply(x1=x1[None, :], x2=x2[:, None])
                .swapaxes(-1, -2)
            )

        calc = lsl.Calc(_evaluate_kernel, **kwargs).update()

        super().__init__(calc, name=name)
        self.update()


class ParamPredictivePointProcessGP(lsl.Var):
    def __init__(
        self,
        locs: Array,
        K: int,
        kernel_cls: type[tfk.AutoCompositeTensorPsdKernel],
        bijector: tfb.Bijector = tfb.Identity(),
        name: str = "",
        **kernel_params: lsl.Var | TransformedVar,
    ) -> None:

        kernel_uu = Kernel(
            x1=locs[:K, :],
            x2=locs[:K, :],
            kernel_cls=kernel_cls,
            **kernel_params,
            name=f"{name}_kernel",
        )

        kernel_du = Kernel(
            x1=locs[K:, :],
            x2=locs[:K, :],
            kernel_cls=kernel_cls,
            **kernel_params,
            name=f"{name}_kernel_latent_u",
        )

        latent_var = lsl.param(
            jnp.zeros((K,)),
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name=f"{name}_latent",
        )

        # a small value added to the diagonal of Kuu for numerical stability
        salt = jnp.diag(jnp.full(shape=(kernel_du.valu.shape[0],), fill_value=1e-6))

        def _compute_param(latent_var, Kuu, Kdu):
            Kuu = Kuu + salt
            L = jnp.linalg.cholesky(Kuu)
            Li = jnp.linalg.inv(L)

            u = L @ latent_var
            alpha = Kdu @ Li.T @ latent_var
            return bijector.forward(jnp.r_[u, alpha])

        super().__init__(
            lsl.Calc(_compute_param, latent_var, kernel_uu, kernel_du),
            name=name,
        )

        self.parameter_names = [latent_var.name]
        self.hyperparameter_names = [
            find_param(param).name for param in kernel_params.values()
        ]


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
        locs: Array,
        D: int,
        K: int,
        kernel_cls: type[tfk.AutoCompositeTensorPsdKernel],
        name: str = "",
        **kernel_params: lsl.Var | TransformedVar,
    ):
        kernel_uu = Kernel(
            x1=locs[:K, :],
            x2=locs[:K, :],
            kernel_cls=kernel_cls,
            **kernel_params,
            name=f"{name}_kernel",
        )

        kernel_du = Kernel(
            x1=locs[K:, :],
            x2=locs[:K, :],
            kernel_cls=kernel_cls,
            **kernel_params,
            name=f"{name}_kernel_latent_u",
        )

        W = rw_weight_matrix(D)

        nrow_W = W.shape[0]

        latent_var = lsl.param(
            jnp.zeros((K * (W.shape[1]),)),
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name=f"{name}_latent",
        )

        def _compute_param(latent_var, Kuu, Kdu):
            salt = jnp.diag(jnp.full(shape=(Kuu.shape[0],), fill_value=1e-6))
            Kuu = Kuu + salt
            L = jnp.linalg.cholesky(Kuu)
            Li = jnp.linalg.inv(L)

            Wkron = jnp.kron(W, L)
            u = Wkron @ latent_var
            u_mat = jnp.reshape(u, (nrow_W, K))

            Kdu_uui = jnp.kron(W, (Kdu @ Li.T))
            delta_long = Kdu_uui @ latent_var
            delta_mat = jnp.reshape(delta_long, (nrow_W, locs.shape[0] - K))

            return jnp.concatenate([u_mat, delta_mat], axis=1)

        super().__init__(
            lsl.Calc(
                _compute_param,
                latent_var=latent_var,
                Kuu=kernel_uu,
                Kdu=kernel_du,
            ),
            name=name,
        )

        self.parameter_names = [latent_var.name]
        self.hyperparameter_names = [
            find_param(param).name for param in kernel_params.values()
        ]


class OnionCoefPredictivePointProcessGP(lsl.Var):
    def __init__(
        self,
        latent_coef: RandomWalkParamPredictivePointProcessGP,
        coef_spec: OnionCoef,
        name: str = "",
    ) -> None:

        super().__init__(
            lsl.Calc(lambda latent: coef_spec(latent.T).T, latent_coef).update(),
            name=name,
        )

        self.parameter_names = latent_coef.parameter_names
        self.hyperparameter_names = latent_coef.hyperparameter_names

    @classmethod
    def new_from_locs(
        cls,
        knots: OnionKnots,
        locs: Array,
        K: int,
        kernel_cls: type[tfk.AutoCompositeTensorPsdKernel],
        name: str = "",
        **kernel_params: lsl.Var | TransformedVar,
    ) -> OnionCoefPredictivePointProcessGP:
        
        coef_spec = OnionCoef(knots)

        latent_coef = RandomWalkParamPredictivePointProcessGP(
            locs=locs, D=knots.nparam + 1, K=K, kernel_cls=kernel_cls, name=f"{name}_log_increments", **kernel_params
        )

        return cls(coef_spec=coef_spec, latent_coef=latent_coef, name=name)
