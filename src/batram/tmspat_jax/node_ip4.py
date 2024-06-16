"""
Nodes for inducing points version.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import optax
import jax.numpy as jnp
import liesel.model as lsl
import liesel_ptm as ptm
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
from liesel.goose.types import ModelState
from liesel_ptm.ptm_ls import NormalizationFn
from .optim import optim_flat, OptimResult

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

def rw_weight_matrix_noncentered(D: int):
    B = brownian_motion_mat(D - 2, D - 2)
    L = jnp.linalg.cholesky(B, upper=False)
    W = jnp.r_[jnp.zeros((2, D - 2)), L]
    W = jnp.c_[jnp.ones((D, 1)), W]
    W = jnp.c_[jnp.zeros((D, 1)), W]
    W = W.at[0,0].set(1.0)
    return W


class DeltaParam(lsl.Var):
    def __init__(
        self,
        locs: Array,
        D: int,
        eta: lsl.Var,
        K: int,
        kernel_class: type[
            tfk.AutoCompositeTensorPsdKernel
        ] = tfk.ExponentiatedQuadratic,
    ):
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
            jnp.zeros((K * D,)),
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

        # W = rw_weight_matrix_noncentered(D)
        W = jnp.eye(D)

        intercept_mean = lsl.param(0.0, name="intercept_mean")
        scale_mean = lsl.param(0.0, name="scale_mean")

        def _compute_delta(intercept_mean, scale_mean, latent_delta, eta, Kuu, Kdu):
            salt = jnp.diag(jnp.full(shape=(Kuu.shape[0],), fill_value=1e-6))
            Kuu = Kuu + salt
            L = jnp.linalg.cholesky(Kuu)
            Li = jnp.linalg.inv(L)

            exp_eta = jnp.expand_dims(jnp.exp(eta), -1)
            Wkron = jnp.kron(W, exp_eta[:K, :] * L)
            u = Wkron @ latent_delta
            u_mat = jnp.reshape(u, (D, K))

            Kdu_uui = jnp.kron(W, exp_eta[K:, :] * (Kdu @ Li.T))
            delta_long = Kdu_uui @ latent_delta
            delta_mat = jnp.reshape(delta_long, (D, locs.shape[0] - K))

            delta = jnp.concatenate([u_mat, delta_mat], axis=1)
            delta = delta.at[0,:].set(delta[0,:] + intercept_mean)
            delta = delta.at[1,:].set(delta[1,:] + scale_mean)
            return delta

        super().__init__(
            lsl.Calc(
                _compute_delta,
                intercept_mean=intercept_mean,
                scale_mean=scale_mean,
                latent_delta=latent_delta,
                eta=eta,
                Kuu=kernel_uu,
                Kdu=kernel_du,
            ),
            name="delta",
        )

        self.parameter_names = [latent_delta.name]
        self.hyperparameter_names = [
            intercept_mean.name,
            scale_mean.name,
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
    def __init__(self, delta: lsl.Var) -> None:
        def _compute_shape_coef(delta):
            exp_delta = delta
            exp_delta = exp_delta.at[1:,:].set(jnp.exp(delta[1:,:]))
            return exp_delta.cumsum(axis=0).T

        super().__init__(lsl.Calc(_compute_shape_coef, delta), name="shape_coef")


class AlphaParam(lsl.Var):
    def __init__(
        self,
        locs: Array,
        knots: Array,
        K: int,
        name: str = "alpha",
        kernel_class: type[
            tfk.AutoCompositeTensorPsdKernel
        ] = tfk.ExponentiatedQuadratic,
    ) -> None:
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

        constant = lsl.param(0.0, name=f"{name}_mean")

        def _compute_param(constant, latent_alpha, Kuu, Kdu):
            salt = jnp.diag(jnp.full(shape=(Kuu.shape[0],), fill_value=1e-6))
            Kuu = Kuu + salt
            L = jnp.linalg.cholesky(Kuu)
            Li = jnp.linalg.inv(L)

            u = L @ latent_alpha
            alpha = Kdu @ Li.T @ latent_alpha
            return constant + jnp.r_[u, alpha] - a

        super().__init__(
            lsl.Calc(_compute_param, constant, latent_alpha, kernel_uu, kernel_du), name=name
        )
        self.parameter_names = [latent_alpha.name]
        self.hyperparameter_names = [
            constant.name,
            amplitude_transformed.name,
            length_scale_transformed.name,
        ]


class EtaParam(lsl.Var):
    def __init__(
        self,
        locs: Array,
        K: int,
        name: str = "eta",
        kernel_class: type[
            tfk.AutoCompositeTensorPsdKernel
        ] = tfk.ExponentiatedQuadratic,
    ) -> None:
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

        mean = lsl.param(-1.0, name=f"{name}_mean")

        def _compute_param(latent, Kuu, Kdu, mean):
            salt = jnp.diag(jnp.full(shape=(Kuu.shape[0],), fill_value=1e-6))
            Kuu = Kuu + salt
            L = jnp.linalg.cholesky(Kuu)
            Li = jnp.linalg.inv(L)

            u = L @ latent
            var = Kdu @ Li.T @ latent
            return jnp.r_[u, var] + mean

        super().__init__(
            lsl.Calc(_compute_param, latent, kernel_uu, kernel_du, mean),
            name=f"{name}",
        )

        self.parameter_names = [latent.name]
        self.hyperparameter_names = [
            amplitude_transformed.name,
            length_scale_transformed.name,
            mean.name
        ]


class EtaParamFixed(lsl.Var):
    def __init__(
        self,
        locs: Array,
        name: str = "eta",
    ) -> None:
        super().__init__(
            value=jnp.full(shape=(locs.shape[0],), fill_value=-1.0),
            name=name,
        )
        self.parameter_names = []
        self.hyperparameter_names = []


class TransformationCoef(lsl.Var):
    """Dimension (Nloc, D)"""

    def __init__(self, alpha: lsl.Var, shape_coef: lsl.Var) -> None:
        def _assemble_trafo_coef(alpha, shape_coef):
            alpha = jnp.expand_dims(alpha, 0)
            scaled_shape = alpha + shape_coef
            coef = jnp.r_[alpha, scaled_shape]
            return coef.T

        super().__init__(
            lsl.Calc(_assemble_trafo_coef, alpha, shape_coef),
            name="trafo_coef",
        )


class Model:
    def __init__(
        self,
        y: Array,
        knots: Array,
        locs: Array,
        K: int,
        kernel_class: type[
            tfk.AutoCompositeTensorPsdKernel
        ] = tfk.ExponentiatedQuadratic,
        extrap_transition_width: float = 0.3,
        eta_fixed: bool = False
    ) -> None:
        D = jnp.shape(knots)[0] - 4
        self.knots = knots
        self.nparam = D
        self.eta_fixed = eta_fixed
        if eta_fixed:
            self.eta = EtaParamFixed(locs=locs)
        else:
            self.eta = EtaParam(locs, K=K, kernel_class=kernel_class).update()
        self.delta = DeltaParam(
            locs, D, self.eta, K=K, kernel_class=kernel_class
        ).update()
        self.coef = ShapeCoef(self.delta).update()

        self.extrap_transition_width = extrap_transition_width
        self.bspline = ptm.ExtrapBSplineApprox(
            knots=knots, order=3, eps=extrap_transition_width
        )

        basis_dot_and_deriv_fn = self.bspline.get_extrap_basis_dot_and_deriv_fn(
            target_slope=1.0
        )

        self.response_value = lsl.obs(y, name="response_hidden_value")

        self.normalization_and_deriv = lsl.Var(
            lsl.Calc(
                lambda y, c: basis_dot_and_deriv_fn(y.T, c),
                self.response_value,
                self.coef,
            ),
            name="normalization_and_deriv",
        ).update()

        self.normalization = lsl.Var(
            lsl.Calc(lambda x: x[0].T, self.normalization_and_deriv),
            name="normalization",
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
        self.response = lsl.obs(y, response_dist, name="response").update()
        """Response variable."""

        self.graph = None

    @classmethod
    def from_nparam(
        cls,
        y: Array,
        locs: Array,
        nparam: int,
        knots_lo: float,
        knots_hi: float,
        K: int,
        kernel_class: type[
            tfk.AutoCompositeTensorPsdKernel
        ] = tfk.ExponentiatedQuadratic,
        extrap_transition_width: float = 0.3,
        eta_fixed: bool = False
    ) -> Model:
        knots = ptm.kn(jnp.array([knots_lo, knots_hi]), order=3, n_params=nparam)
        return cls(
            y=y,
            knots=knots,
            locs=locs,
            K=K,
            kernel_class=kernel_class,
            extrap_transition_width=extrap_transition_width,
            eta_fixed=eta_fixed
        )

    def build_graph(self):
        self.graph = lsl.GraphBuilder().add(self.response).build_model()
        return self.graph
    
    def param(self) -> list[str]:
        param = (
            self.delta.parameter_names
        )

        if not self.eta_fixed:
            param += self.eta.parameter_names
        
        return param
    
    
    def hyperparam(self) -> list[str]:
        hyperparam = (
            self.delta.hyperparameter_names
        )

        if not self.eta_fixed:
            hyperparam += self.eta.hyperparameter_names
        
        return hyperparam

    
    def fit(
        self,
        graph: lsl.Model | None = None,
        graph_validation: lsl.Model | None = None,
        optimizer: optax.GradientTransformation | None = None,
        stopper: ptm.Stopper | None = None
    ) -> OptimResult:
        graph = self.graph if graph is None else graph
        param = self.param()
        hyperparam = self.hyperparam()

        result = optim_flat(
            graph,
            params=param + hyperparam,
            stopper=stopper,
            optimizer=optimizer,
            model_validation=graph_validation,
        )
        graph.state = result.model_state
        graph.update()
        return result
    
    def normalization_and_logdet(self, y: Array) -> tuple[Array, Array]:
        _, vars_ = self.graph.copy_nodes_and_vars()
        graph_copy = lsl.GraphBuilder().add(vars_["response"]).build_model()
        graph_copy.vars["response_hidden_value"].value = y
        graph_copy.update()
        z = graph_copy.vars["normalization"].value
        logdet = jnp.log(graph_copy.vars["normalization_deriv"].value)
        return z, logdet
    
    def normalization_inverse(self, z: Array) -> Array:
        hfn = NormalizationFn(
            knots=self.knots, order=3, transition_width=self.extrap_transition_width
        )

        y = hfn.inverse(
            z=z.T, coef=self.coef.update().value, norm_mean=jnp.zeros(1), norm_sd=jnp.ones(1)
        ).T

        return y


def predict_normalization_and_deriv(
    graph: lsl.Model, y: Array, model_state: ModelState
) -> Array:
    """
    y: (Nobs, Nloc)
    """
    graph.state = model_state
    graph.vars["response_hidden_value"].value = y
    graph.update()
    return graph.vars["normalization"].value, graph.vars["normalization_deriv"].value


def predict_normalization_inverse(
    z: Array, coef: Array, model: Model, ngrid: int = 200
):
    hfn = NormalizationFn(
        knots=model.knots, order=3, transition_width=model.extrap_transition_width
    )

    return hfn.inverse(
        z=z.T, coef=coef, norm_mean=jnp.zeros(1), norm_sd=jnp.ones(1), ngrid=ngrid
    ).T
