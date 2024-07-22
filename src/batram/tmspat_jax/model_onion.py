"""
Nodes for inducing points version.

- Alpha parameter: Intercept with its own hyperparameters
- Alpha parameter: Has prior mean, is estimated
- Slope has prior mean, is estimated
"""

from __future__ import annotations

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
from liesel_ptm.bsplines import OnionCoef, Knots, StreamCoef, SimpleOnionCoef, SimpleStreamCoef
from .optim import optim_flat, OptimResult
from enum import Enum, auto

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
    W = jnp.r_[jnp.zeros((1, D - 2)), L]
    W = jnp.c_[jnp.ones((D - 1, 1)), W]
    return W


def weighted_moving_average_matrix(n, weights):
    k = len(weights)
    wma_matrix = jnp.zeros((n, n))

    for i in range(n):
        for j in range(max(0, i - k + 1), i + 1):
            wma_matrix = wma_matrix.at[i, j].set(weights[k - (i - j) - 1])

    # Normalize the matrix rows to ensure weighted average
    for i in range(n):
        row_sum = jnp.sum(wma_matrix[i])
        if row_sum != 0:
            wma_matrix = wma_matrix.at[i].set(wma_matrix[i] / row_sum)

    return wma_matrix


class DeltaSmoothing(Enum):
    RANDOM_WALK = auto()
    RIDGE = auto()
    RIDGE_MOVING_AVERAGE = auto()


class DeltaParam(lsl.Var):
    def __init__(
        self,
        locs: Array,
        D: int,
        eta: lsl.Var,
        K: int,
        smoothing_prior: DeltaSmoothing,
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

        self.amplitude = amplitude
        self.length_scale = length_scale

        

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

        if smoothing_prior == DeltaSmoothing.RANDOM_WALK:
            W = rw_weight_matrix(D)
        elif smoothing_prior == DeltaSmoothing.RIDGE:
            W = jnp.eye(D - 1)
        elif smoothing_prior == DeltaSmoothing.RIDGE_MOVING_AVERAGE:
            W = weighted_moving_average_matrix(n=(D - 1), weights=jnp.array([0.5, 0.5]))
        else:
            raise ValueError(f"'{smoothing_prior=}': value is not recognized.")

        nrow_W = W.shape[0]

        latent_delta = lsl.param(
            jnp.zeros((K * (W.shape[1]),)),
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="latent_delta",
        )

        def _compute_delta(latent_delta, eta, Kuu, Kdu):
            salt = jnp.diag(jnp.full(shape=(Kuu.shape[0],), fill_value=1e-6))
            Kuu = Kuu + salt
            L = jnp.linalg.cholesky(Kuu)
            Li = jnp.linalg.inv(L)

            exp_eta = jnp.expand_dims(jnp.exp(eta), -1)
            Wkron = jnp.kron(W, exp_eta[:K, :] * L)
            u = Wkron @ latent_delta
            u_mat = jnp.reshape(u, (nrow_W, K))

            Kdu_uui = jnp.kron(W, exp_eta[K:, :] * (Kdu @ Li.T))
            delta_long = Kdu_uui @ latent_delta
            delta_mat = jnp.reshape(delta_long, (nrow_W, locs.shape[0] - K))

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


class AlphaParam(lsl.Var):
    def __init__(
        self,
        locs: Array,
        K: int,
        name: str = "alpha",
        kernel_class: type[
            tfk.AutoCompositeTensorPsdKernel
        ] = tfk.ExponentiatedQuadratic,
        amplitude: lsl.Var | None = None,
        length_scale: lsl.Var | None = None,
    ) -> None:
        kernel_args = dict()

        self.hyperparameter_names = []

        if amplitude is None:
            amplitude_transformed = lsl.param(0.5, name=f"amplitude_{name}_transformed")
            amplitude = lsl.Var(
                lsl.Calc(jax.nn.softplus, amplitude_transformed),
                name=f"amplitude_{name}",
            )
            self.hyperparameter_names.append(amplitude.name)

        if length_scale is None:
            length_scale_transformed = lsl.param(
                0.5, name=f"length_scale_{name}_transformed"
            )
            length_scale = lsl.Var(
                lsl.Calc(jax.nn.softplus, length_scale_transformed),
                name=f"length_scale_{name}",
            )
            self.hyperparameter_names.append(length_scale.name)

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

        constant = lsl.param(0.0, name=f"{name}_mean")

        def _compute_param(constant, latent_alpha, Kuu, Kdu):
            salt = jnp.diag(jnp.full(shape=(Kuu.shape[0],), fill_value=1e-6))
            Kuu = Kuu + salt
            L = jnp.linalg.cholesky(Kuu)
            Li = jnp.linalg.inv(L)

            u = L @ latent_alpha
            alpha = Kdu @ Li.T @ latent_alpha
            return constant + jnp.r_[u, alpha]

        super().__init__(
            lsl.Calc(_compute_param, constant, latent_alpha, kernel_uu, kernel_du),
            name=name,
        )
        self.parameter_names = [latent_alpha.name]
        self.hyperparameter_names.append(constant.name)


class ExpBetaParam(lsl.Var):
    def __init__(
        self,
        locs: Array,
        K: int,
        name: str = "beta",
        kernel_class: type[
            tfk.AutoCompositeTensorPsdKernel
        ] = tfk.ExponentiatedQuadratic,
        amplitude: lsl.Var | None = None,
        length_scale: lsl.Var | None = None,
    ) -> None:
        kernel_args = dict()
        self.hyperparameter_names = []

        if amplitude is None:
            amplitude_transformed = lsl.param(0.5, name=f"amplitude_{name}_transformed")
            amplitude = lsl.Var(
                lsl.Calc(jax.nn.softplus, amplitude_transformed),
                name=f"amplitude_{name}",
            )
            self.hyperparameter_names.append(amplitude.name)

        if length_scale is None:
            length_scale_transformed = lsl.param(
                0.5, name=f"length_scale_{name}_transformed"
            )
            length_scale = lsl.Var(
                lsl.Calc(jax.nn.softplus, length_scale_transformed),
                name=f"length_scale_{name}",
            )
            self.hyperparameter_names.append(length_scale.name)

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

        constant = lsl.param(0.0, name=f"{name}_mean")

        def _compute_param(constant, latent, Kuu, Kdu):
            salt = jnp.diag(jnp.full(shape=(Kuu.shape[0],), fill_value=1e-6))
            Kuu = Kuu + salt
            L = jnp.linalg.cholesky(Kuu)
            Li = jnp.linalg.inv(L)

            u = L @ latent
            var = Kdu @ Li.T @ latent
            return jnp.exp(jnp.r_[u, var] + constant)

        super().__init__(
            lsl.Calc(_compute_param, constant, latent_beta, kernel_uu, kernel_du),
            name=f"exp_{name}",
        )
        self.parameter_names = [latent_beta.name]
        self.hyperparameter_names.append(constant.name)


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
            mean.name,
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

    def __init__(
        self,
        alpha: lsl.Var | None,
        exp_beta: lsl.Var | None,
        shape_coef: lsl.Var,
        coef_spec: OnionCoef | StreamCoef | SimpleOnionCoef | SimpleStreamCoef,
    ) -> None:

        alpha = (
            alpha
            if alpha is not None
            else lsl.Data(jnp.zeros(1), _name="auto_alpha_fixed_to_zero")
        )
        exp_beta = (
            exp_beta
            if exp_beta is not None
            else lsl.Data(jnp.ones(1), _name="auto_exp_beta_fixed_to_one")
        )

        def _assemble_trafo_coef(alpha, exp_beta, shape_coef):
            alpha = jnp.expand_dims(alpha, 0)
            exp_beta = jnp.expand_dims(exp_beta, 0)
            coef = coef_spec.compute_coef(shape_coef.T).T
            coef = alpha + exp_beta * coef
            return coef.T

        super().__init__(
            lsl.Calc(_assemble_trafo_coef, alpha, exp_beta, shape_coef),
            name="trafo_coef",
        )


class Model:
    def __init__(
        self,
        y: Array,
        knots: Knots,
        coef_spec: OnionCoef | StreamCoef,
        locs: Array,
        K: int,
        smoothing_prior: DeltaSmoothing,
        kernel_class: type[
            tfk.AutoCompositeTensorPsdKernel
        ] = tfk.ExponentiatedQuadratic,
        extrap_transition_width: float = 0.3,
        eta_fixed: bool = False,
        include_alpha: bool = False,
        include_exp_beta: bool = False,
        shared_hyperparameters_alpha: bool = True,
        shared_hyperparameters_beta: bool = True,
    ) -> None:
        self.knots = knots
        self.eta_fixed = eta_fixed

        if eta_fixed:
            self.eta = EtaParamFixed(locs=locs)
        else:
            self.eta = EtaParam(locs, K=K, kernel_class=kernel_class).update()
        self.delta = DeltaParam(
            locs, self.knots.nparam + 1, self.eta, K=K, kernel_class=kernel_class,
            smoothing_prior=smoothing_prior
        ).update()

        if include_alpha and not shared_hyperparameters_alpha:
            self.alpha = AlphaParam(locs, K=K, kernel_class=kernel_class).update()
        elif include_alpha and shared_hyperparameters_alpha:
            self.alpha = AlphaParam(
                locs,
                K=K,
                kernel_class=kernel_class,
                length_scale=self.delta.length_scale,
                amplitude=self.delta.amplitude,
            ).update()
        else:
            self.alpha = None

        if include_exp_beta and not shared_hyperparameters_beta:
            self.exp_beta = ExpBetaParam(locs, K=K, kernel_class=kernel_class).update()
        elif include_exp_beta and shared_hyperparameters_beta:
            self.exp_beta = ExpBetaParam(
                locs,
                K=K,
                kernel_class=kernel_class,
                length_scale=self.delta.length_scale,
                amplitude=self.delta.amplitude,
            ).update()
        else:
            self.exp_beta = None

        self.coef = TransformationCoef(
            self.alpha, self.exp_beta, self.delta, coef_spec=coef_spec
        ).update()

        self.extrap_transition_width = extrap_transition_width
        self.bspline = ptm.ExtrapBSplineApprox(
            knots=knots.knots, order=3, eps=extrap_transition_width
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

    def build_graph(self):
        self.graph = lsl.GraphBuilder().add(self.response).build_model()
        return self.graph

    def param(self) -> list[str]:
        param = self.delta.parameter_names

        if not self.eta_fixed:
            param += self.eta.parameter_names

        if self.alpha is not None:
            param += self.alpha.parameter_names

        if self.exp_beta is not None:
            param += self.exp_beta.parameter_names

        return param

    def hyperparam(self) -> list[str]:
        hyperparam = self.delta.hyperparameter_names

        if not self.eta_fixed:
            hyperparam += self.eta.hyperparameter_names

        if self.alpha is not None:
            hyperparam += self.alpha.hyperparameter_names

        if self.exp_beta is not None:
            hyperparam += self.exp_beta.hyperparameter_names

        return hyperparam

    def fit(
        self,
        graph: lsl.Model | None = None,
        graph_validation: lsl.Model | None = None,
        optimizer: optax.GradientTransformation | None = None,
        stopper: ptm.Stopper | None = None,
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
            knots=self.knots.knots,
            order=3,
            transition_width=self.extrap_transition_width,
        )

        y = hfn.inverse(
            z=z.T,
            coef=self.coef.update().value,
            norm_mean=jnp.zeros(1),
            norm_sd=jnp.ones(1),
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
        knots=model.knots.knots, order=3, transition_width=model.extrap_transition_width
    )

    return hfn.inverse(
        z=z.T, coef=coef, norm_mean=jnp.zeros(1), norm_sd=jnp.ones(1), ngrid=ngrid
    ).T
