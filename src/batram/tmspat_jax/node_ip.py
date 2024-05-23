"""
Nodes for inducing points version.
"""

from __future__ import annotations
from functools import partial
from typing import Any


import jax.numpy as jnp
import liesel.model as lsl
import liesel_ptm as ptm
import jax
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
from liesel.goose.types import ModelState

Array = Any


class Kernel(lsl.Var):
    def __init__(
        self,
        x: Array,
        kernel_class: type[tfk.AutoCompositeTensorPsdKernel],
        name: str = "",
        **kwargs,
    ) -> None:
        if not jnp.ndim(x) == 2:
            raise ValueError("x must be a 2d array.")

        self.x = x
        self.kernel_class = kernel_class

        def _evaluate_kernel(**kwargs):
            return kernel_class(**kwargs).apply(x1=x[None, :], x2=x[:, None])

        calc = lsl.Calc(_evaluate_kernel, **kwargs)

        super().__init__(calc, name=name)


class Kernel2(lsl.Var):
    def __init__(
        self,
        x1: Array,
        x2: Array,
        kernel_class: type[tfk.AutoCompositeTensorPsdKernel],
        name: str = "",
        **kwargs,
    ) -> None:
        if not jnp.ndim(x1) == 2:
            raise ValueError("x1 must be a 2d array.")
        if not jnp.ndim(x2) == 2:
            raise ValueError("x2 must be a 2d array.")

        self.x1 = x1
        self.x2 = x2
        self.kernel_class = kernel_class

        def _evaluate_kernel(**kwargs):
            return kernel_class(**kwargs).apply(x1=x1[:, None], x2=x2[None, :])

        calc = lsl.Calc(_evaluate_kernel, **kwargs)

        super().__init__(calc, name=name)


class MultioutputKernelIMC(lsl.Var):
    def __init__(
        self,
        x: Array,
        W: Array,
        kernel_class: type[tfk.AutoCompositeTensorPsdKernel],
        name: str = "",
        **kwargs,
    ) -> None:
        if not jnp.ndim(x) == 2:
            raise ValueError("x must be a 2d array.")

        self.x = x
        self.kernel_class = kernel_class
        self.W = W

        def _evaluate_kernel(**kwargs):
            kernel = kernel_class(**kwargs).apply(x1=x[None, :], x2=x[:, None])

            ID = jnp.eye(W.shape[1])
            kernel_blockdiagonal = jnp.kron(ID, kernel)

            IN = jnp.eye(x.shape[0])
            W_kron = jnp.kron(W, IN)

            return W_kron @ kernel_blockdiagonal @ W_kron.T

        calc = lsl.Calc(_evaluate_kernel, **kwargs)

        super().__init__(calc, name=name)


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


class DeltaParam(lsl.Var):
    def __init__(self, locs: Array, D: int, eta: lsl.Var, K: int, kernel_class: type[tfk.AutoCompositeTensorPsdKernel] = tfk.ExponentiatedQuadratic):
        kernel_args = dict()

        amplitude_transformed = lsl.param(0.5, name="amplitude_delta_transformed")
        amplitude = lsl.Var(
            lsl.Calc(jax.nn.softplus, amplitude_transformed), name="amplitude_delta"
        )

        length_scale_transformed = lsl.param(0.5, name="length_scale_delta_transformed")
        length_scale = lsl.Var(
            lsl.Calc(jax.nn.softplus, length_scale_transformed),
            name="length_scale_delta",
        )

        kernel_args["amplitude"] = amplitude
        kernel_args["length_scale"] = length_scale

        latent_delta = lsl.param(
            jnp.zeros((K * (D - 2),)),
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="latent_delta",
        )

        kernel_uu = Kernel(
            x=locs[:K, :],
            kernel_class=kernel_class,
            **kernel_args,
            name="kernel_latent_delta",
        )

        kernel_du = Kernel2(
            x1=locs[K:, :],
            x2=locs[:K, :],
            kernel_class=kernel_class,
            **kernel_args,
            name="kernel_latent_delta_u",
        )

        W = rw_weight_matrix(D)

        def _compute_delta(latent_delta, eta, Kuu, Kdu):
            salt = jnp.diag(jnp.full(shape=(Kuu.shape[0],), fill_value=1e-6))
            Kuu = Kuu + salt
            L = jnp.linalg.cholesky(Kuu)
            Li = jnp.linalg.inv(L)

            exp_eta = jnp.expand_dims(jnp.exp(eta), -1)
            Wkron = jnp.kron(W, exp_eta[:K, :] * L)
            u = Wkron @ latent_delta
            u_mat = jnp.reshape(u, (D - 1, K))

            Kdu_uui = jnp.kron(W, exp_eta[K:, :] * (Kdu @ Li.T))
            delta_long = Kdu_uui @ latent_delta
            delta_mat = jnp.reshape(delta_long, (D - 1, locs.shape[0] - K))

            return jnp.concatenate([u_mat, delta_mat], axis=1)

        super().__init__(
            lsl.Calc(
                _compute_delta,
                latent_delta=latent_delta,
                eta=eta,
                Kuu=kernel_uu,
                Kdu=kernel_du,
            ),
            name="delta",
        )

        self.parameter_names = [latent_delta.name]
        self.hyperparameter_names = [
            amplitude_transformed.name,
            length_scale_transformed.name,
        ]

        self.latent = latent_delta


@partial(jnp.vectorize, excluded=[1], signature="(d)->()")
def sfn(exp_shape, dknots: float | Array):
    order = 3
    p = jnp.shape(exp_shape)[-1] + 1

    outer_border = exp_shape[..., jnp.array([0, -1])] / 6
    inner_border = 5 * exp_shape[..., jnp.array([1, -2])] / 6
    middle = exp_shape[..., 2:-2]
    summed_exp_shape = (
        outer_border.sum(axis=-1, keepdims=True)
        + inner_border.sum(axis=-1, keepdims=True)
        + middle.sum(axis=-1, keepdims=True)
    )

    return jnp.squeeze((1 / ((p - order) * dknots)) * summed_exp_shape)


class ShapeCoef(lsl.Var):
    def __init__(self, delta: lsl.Var, dknots: float | Array) -> None:
        def _compute_shape_coef(delta):
            exp_delta = jnp.exp(delta)
            slope_correction_factor = sfn(exp_delta.T, dknots)
            return exp_delta.cumsum(axis=0) / jnp.expand_dims(
                slope_correction_factor, 0
            )

        super().__init__(lsl.Calc(_compute_shape_coef, delta), name="shape_coef")


class AlphaParam(lsl.Var):
    def __init__(self, locs: Array, knots: Array, K: int, name: str = "alpha", kernel_class: type[tfk.AutoCompositeTensorPsdKernel] = tfk.ExponentiatedQuadratic) -> None:
        kernel_args = dict()

        amplitude_transformed = lsl.param(0.5, name=f"amplitude_{name}_transformed")
        amplitude = lsl.Var(
            lsl.Calc(jax.nn.softplus, amplitude_transformed), name=f"amplitude_{name}"
        )

        length_scale_transformed = lsl.param(
            0.5, name=f"length_scale_{name}_transformed"
        )
        length_scale = lsl.Var(
            lsl.Calc(jax.nn.softplus, length_scale_transformed),
            name=f"length_scale_{name}",
        )

        kernel_args["amplitude"] = amplitude
        kernel_args["length_scale"] = length_scale
        kernel_uu = Kernel(
            x=locs[:K, :],
            kernel_class=kernel_class,
            **kernel_args,
            name=f"kernel_{name}",
        )

        kernel_du = Kernel2(
            x1=locs[K:, :],
            x2=locs[:K, :],
            kernel_class=kernel_class,
            **kernel_args,
            name=f"kernel_latent_{name}_u",
        )

        latent_alpha = lsl.param(
            jnp.zeros((K,)),
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name=f"latent_{name}",
        )

        a = knots[4] - 2 * knots[3]

        def _compute_param(latent_alpha, Kuu, Kdu):
            salt = jnp.diag(jnp.full(shape=(Kuu.shape[0],), fill_value=1e-6))
            Kuu = Kuu + salt
            L = jnp.linalg.cholesky(Kuu)
            Li = jnp.linalg.inv(L)

            u = L @ latent_alpha
            alpha = Kdu @ Li.T @ latent_alpha
            return jnp.r_[u, alpha] - a

        super().__init__(
            lsl.Calc(_compute_param, latent_alpha, kernel_uu, kernel_du), name=name
        )
        self.parameter_names = [latent_alpha.name]
        self.hyperparameter_names = [
            amplitude_transformed.name,
            length_scale_transformed.name,
        ]


class ExpBetaParam(lsl.Var):
    def __init__(self, locs: Array, K: int, name: str = "beta", kernel_class: type[tfk.AutoCompositeTensorPsdKernel] = tfk.ExponentiatedQuadratic) -> None:
        kernel_args = dict()
        amplitude_transformed = lsl.param(0.5, name=f"amplitude_{name}_transformed")
        amplitude = lsl.Var(
            lsl.Calc(jax.nn.softplus, amplitude_transformed), name=f"amplitude_{name}"
        )
        length_scale_transformed = lsl.param(
            0.5, name=f"length_scale_{name}_transformed"
        )
        length_scale = lsl.Var(
            lsl.Calc(jax.nn.softplus, length_scale_transformed),
            name=f"length_scale_{name}",
        )
        kernel_args["amplitude"] = amplitude
        kernel_args["length_scale"] = length_scale

        kernel_uu = Kernel(
            x=locs[:K, :],
            kernel_class=kernel_class,
            **kernel_args,
            name=f"kernel_{name}",
        )

        kernel_du = Kernel2(
            x1=locs[K:, :],
            x2=locs[:K, :],
            kernel_class=kernel_class,
            **kernel_args,
            name=f"kernel_latent_{name}_u",
        )

        latent_beta = lsl.param(
            jnp.zeros((K,)),
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name=f"latent_{name}",
        )

        def _compute_param(latent, Kuu, Kdu):
            salt = jnp.diag(jnp.full(shape=(Kuu.shape[0],), fill_value=1e-6))
            Kuu = Kuu + salt
            L = jnp.linalg.cholesky(Kuu)
            Li = jnp.linalg.inv(L)

            u = L @ latent
            var = Kdu @ Li.T @ latent
            return jnp.exp(jnp.r_[u, var])

        super().__init__(
            lsl.Calc(_compute_param, latent_beta, kernel_uu, kernel_du),
            name=f"exp_{name}",
        )
        self.parameter_names = [latent_beta.name]
        self.hyperparameter_names = [
            amplitude_transformed.name,
            length_scale_transformed.name,
        ]


class EtaParam(lsl.Var):
    def __init__(self, locs: Array, K: int, name: str = "eta", kernel_class: type[tfk.AutoCompositeTensorPsdKernel] = tfk.ExponentiatedQuadratic) -> None:
        kernel_args = dict()
        amplitude_transformed = lsl.param(0.5, name=f"amplitude_{name}_transformed")
        amplitude = lsl.Var(
            lsl.Calc(jax.nn.softplus, amplitude_transformed), name=f"amplitude_{name}"
        )
        length_scale_transformed = lsl.param(
            0.5, name=f"length_scale_{name}_transformed"
        )
        length_scale = lsl.Var(
            lsl.Calc(jax.nn.softplus, length_scale_transformed),
            name=f"length_scale_{name}",
        )
        kernel_args["amplitude"] = amplitude
        kernel_args["length_scale"] = length_scale

        kernel_uu = Kernel(
            x=locs[:K, :],
            kernel_class=kernel_class,
            **kernel_args,
            name=f"kernel_{name}",
        )

        kernel_du = Kernel2(
            x1=locs[K:, :],
            x2=locs[:K, :],
            kernel_class=kernel_class,
            **kernel_args,
            name=f"kernel_latent_{name}_u",
        )

        latent = lsl.param(
            jnp.zeros((K,)),
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name=f"latent_{name}",
        )

        def _compute_param(latent, Kuu, Kdu):
            salt = jnp.diag(jnp.full(shape=(Kuu.shape[0],), fill_value=1e-6))
            Kuu = Kuu + salt
            L = jnp.linalg.cholesky(Kuu)
            Li = jnp.linalg.inv(L)

            u = L @ latent
            var = Kdu @ Li.T @ latent
            return jnp.r_[u, var]

        super().__init__(
            lsl.Calc(_compute_param, latent, kernel_uu, kernel_du),
            name=f"{name}",
        )

        self.parameter_names = [latent.name]
        self.hyperparameter_names = [
            amplitude_transformed.name,
            length_scale_transformed.name,
        ]


class TransformationCoef(lsl.Var):
    """Dimension (Nloc, D)"""

    def __init__(self, alpha: lsl.Var, exp_beta: lsl.Var, shape_coef: lsl.Var) -> None:
        def _assemble_trafo_coef(alpha, exp_beta, shape_coef):
            alpha = jnp.expand_dims(alpha, 0)
            scaled_shape = alpha + jnp.expand_dims(exp_beta, 0) * shape_coef
            coef = jnp.r_[alpha, scaled_shape]
            return coef.T

        super().__init__(
            lsl.Calc(_assemble_trafo_coef, alpha, exp_beta, shape_coef),
            name="trafo_coef",
        )


class Model:
    def __init__(self, y: Array, knots: Array, locs: Array, K: int, kernel_class: type[tfk.AutoCompositeTensorPsdKernel] = tfk.ExponentiatedQuadratic) -> None:
        D = jnp.shape(knots)[0] - 4
        dknots = jnp.diff(knots).mean()
        self.knots = knots
        self.nparam = D

        self.eta = EtaParam(locs, K=K, kernel_class=kernel_class).update()
        self.delta = DeltaParam(locs, D, self.eta, K=K, kernel_class=kernel_class).update()
        self.cumsum_exp_delta = ShapeCoef(self.delta, dknots).update()
        self.exp_beta = ExpBetaParam(locs, K=K, kernel_class=kernel_class).update()
        self.alpha = AlphaParam(locs, knots, K=K, kernel_class=kernel_class).update()

        self.coef = TransformationCoef(
            self.alpha, self.exp_beta, self.cumsum_exp_delta
        ).update()

        self.bspline = ptm.ExtrapBSplineApprox(knots=knots, order=3)

        basis_dot_and_deriv_fn = self.bspline.get_extrap_basis_dot_and_deriv_fn(
            target_slope=1.0
        )

        self.response_value = lsl.obs(y, name="response_hidden_value")

        self.normalization_and_deriv = lsl.Var(
            lsl.Calc(lambda y, c: basis_dot_and_deriv_fn(y.T, c), self.response_value, self.coef),
            name="normalization_and_deriv",
        ).update()

        self.normalization = lsl.Var(
            lsl.Calc(lambda x: x[0].T, self.normalization_and_deriv), name="normalization"
        ).update()

        self.normalization_deriv = lsl.Var(
            lsl.Calc(lambda x: x[1].T, self.normalization_and_deriv),
            name="normalization_deriv",
        ).update()

        self.refdist = tfd.Normal(loc=0.0, scale=1.0)
        """
        The reference distribution, currently fixed to the standard normal distribution.
        """

        response_dist = ptm.TransformationDist(
            self.normalization, self.normalization_deriv, refdist=self.refdist
        )
        self.response = lsl.obs(
            y, response_dist, name="response"
        ).update()
        """Response variable."""

    @classmethod
    def from_nparam(
        cls, y: Array, locs: Array, nparam: int, knots_lo: float, knots_hi: float, K: int
    ) -> Model:
        knots = ptm.kn(jnp.array([knots_lo, knots_hi]), order=3, n_params=nparam)
        return cls(y=y, knots=knots, locs=locs, K=K)

    def build_graph(self):
        graph = lsl.GraphBuilder().add(self.response).build_model()
        return graph


def predict_normalization_and_deriv(
    graph: lsl.Model, y: Array, model_state: ModelState
) -> Array:
    """
    y: (Nloc, Nobs)
    """
    graph.state = model_state
    graph.vars["response_hidden_value"].value = y
    graph.update()
    return graph.vars["normalization"].value, graph.vars["normalization_deriv"].value
