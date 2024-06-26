from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import liesel.model as lsl
import liesel_ptm as ptm
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


def delta_param(
    locs: Array,
    D: int,
    eta: lsl.Var,
    kernel_class: type[tfk.AutoCompositeTensorPsdKernel] = tfk.ExponentiatedQuadratic,
) -> lsl.Var:
    """
    Dimension: (D-1, Nloc)
    """

    kernel_args = dict()

    amplitude_transformed = lsl.Var(1.0, name="amplitude_delta_transformed")
    amplitude = lsl.Var(
        lsl.Calc(jax.nn.softplus, amplitude_transformed), name="amplitude_delta"
    )
    kernel_args["amplitude"] = amplitude
    kernel_args["length_scale"] = lsl.param(value=1.0, name="length_scale_delta")

    latent_delta = lsl.param(
        jnp.zeros((locs.shape[0] * (D - 2),)),
        distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
        name="latent_delta",
    )

    kernel = Kernel(
        x=locs, kernel_class=kernel_class, **kernel_args, name="kernel_latent_delta"
    )

    def _L_fn(x):
        augmented_x = x + jnp.diag(jnp.full(shape=(x.shape[0],), fill_value=1e-6))
        return jnp.linalg.cholesky(augmented_x)

    Linv = lsl.Var(lsl.Calc(_L_fn, kernel)).update()

    W = rw_weight_matrix(D)

    def _compute_delta(latent_delta, eta, L):
        Wkron = jnp.kron(W, jnp.exp(eta) * L)
        delta_long = Wkron @ latent_delta
        return jnp.reshape(delta_long, (D - 1, locs.shape[0]))

    delta = lsl.Var(
        lsl.Calc(_compute_delta, latent_delta=latent_delta, eta=eta, L=Linv),
        name="delta",
    )

    return delta


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


def shape_coef(delta: lsl.Var, dknots: float | Array) -> lsl.Var:
    def _compute_shape_coef(delta):
        exp_delta = jnp.exp(delta)
        slope_correction_factor = sfn(exp_delta.T, dknots)
        return exp_delta.cumsum(axis=0) / jnp.expand_dims(slope_correction_factor, 0)

    shape_coef = lsl.Var(lsl.Calc(_compute_shape_coef, delta), name="shape_coef")

    return shape_coef


def alpha_param(
    locs: Array,
    knots: Array,
    name: str = "alpha",
    kernel_class: type[tfk.AutoCompositeTensorPsdKernel] = tfk.ExponentiatedQuadratic,
) -> lsl.Var:
    """
    Dimension: (Nloc,)
    """
    kernel_args = dict()
    amplitude_transformed = lsl.Var(1.0, name=f"amplitude_{name}_transformed")
    amplitude = lsl.Var(
        lsl.Calc(jax.nn.softplus, amplitude_transformed), name=f"amplitude_{name}"
    )
    kernel_args["amplitude"] = amplitude
    kernel_args["length_scale"] = lsl.param(value=1.0, name=f"length_scale_{name}")
    kernel = Kernel(
        x=locs,
        kernel_class=kernel_class,
        **kernel_args,
        name=f"kernel_{name}",
    )

    alpha = lsl.param(
        jnp.zeros((locs.shape[0],)),
        distribution=lsl.Dist(
            tfd.MultivariateNormalFullCovariance, covariance_matrix=kernel
        ),
        name=f"{name}",
    )

    a = knots[4] - 2 * knots[3]

    locshift = lsl.Var(lsl.Calc(lambda alpha: alpha - a, alpha), name="locshift")

    return locshift


def beta_param(
    locs: Array,
    name: str = "beta",
    kernel_class: type[tfk.AutoCompositeTensorPsdKernel] = tfk.ExponentiatedQuadratic,
) -> lsl.Var:
    """
    Dimension: (Nloc,)
    """
    kernel_args = dict()
    amplitude_transformed = lsl.Var(1.0, name=f"amplitude_{name}_transformed")
    amplitude = lsl.Var(
        lsl.Calc(jax.nn.softplus, amplitude_transformed), name=f"amplitude_{name}"
    )
    kernel_args["amplitude"] = amplitude
    kernel_args["length_scale"] = lsl.param(value=1.0, name=f"length_scale_{name}")
    kernel = Kernel(
        x=locs,
        kernel_class=kernel_class,
        **kernel_args,
        name=f"kernel_{name}",
    )

    beta = lsl.param(
        jnp.zeros((locs.shape[0],)),
        distribution=lsl.Dist(
            tfd.MultivariateNormalFullCovariance, covariance_matrix=kernel
        ),
        name=f"{name}",
    )

    exp_beta = lsl.Var(lsl.Calc(jnp.exp, beta), name="exp_beta")

    return exp_beta


def eta_param(
    locs: Array,
    name: str = "eta",
    kernel_class: type[tfk.AutoCompositeTensorPsdKernel] = tfk.ExponentiatedQuadratic,
) -> lsl.Var:
    """
    Dimension: (Nloc,)
    """
    kernel_args = dict()
    amplitude_transformed = lsl.Var(1.0, name=f"amplitude_{name}_transformed")
    amplitude = lsl.Var(
        lsl.Calc(jax.nn.softplus, amplitude_transformed), name=f"amplitude_{name}"
    )
    kernel_args["amplitude"] = amplitude
    kernel_args["length_scale"] = lsl.param(value=1.0, name=f"length_scale_{name}")
    kernel = Kernel(
        x=locs,
        kernel_class=kernel_class,
        **kernel_args,
        name=f"kernel_{name}",
    )

    eta = lsl.param(
        jnp.zeros((locs.shape[0],)),
        distribution=lsl.Dist(
            tfd.MultivariateNormalFullCovariance,
            loc=jnp.zeros(locs.shape[0]),
            covariance_matrix=kernel,
        ),
        name=f"{name}",
    )

    return eta


def trafo_coef(alpha: lsl.Var, exp_beta: lsl.Var, shape_coef: lsl.Var) -> lsl.Var:
    """Dimension (Nloc, D)"""

    def _assemble_trafo_coef(alpha, exp_beta, shape_coef):
        alpha = jnp.expand_dims(alpha, 0)
        scaled_shape = alpha + jnp.expand_dims(exp_beta, 0) * shape_coef
        coef = jnp.r_[alpha, scaled_shape]
        return coef.T

    coef = lsl.Var(
        lsl.Calc(_assemble_trafo_coef, alpha, exp_beta, shape_coef), name="trafo_coef"
    )
    return coef


class Model:
    def __init__(
        self,
        y: Array,
        knots: Array,
        locs: Array,
        kernel_class: type[
            tfk.AutoCompositeTensorPsdKernel
        ] = tfk.ExponentiatedQuadratic,
        extrap_transition_width: float = 0.3,
    ) -> None:
        """
        kernel_class must take the arguments ``amplitude`` and ``length_scale``.
        """
        D = jnp.shape(knots)[0] - 4
        dknots = jnp.diff(knots).mean()
        self.knots = knots
        self.nparam = D

        self.eta = eta_param(locs, kernel_class=kernel_class).update()
        self.delta = delta_param(locs, D, self.eta, kernel_class=kernel_class).update()
        self.cumsum_exp_delta = shape_coef(self.delta, dknots).update()
        self.exp_beta = beta_param(locs, kernel_class=kernel_class).update()
        self.alpha = alpha_param(locs, knots, kernel_class=kernel_class).update()

        self.coef = trafo_coef(
            self.alpha, self.exp_beta, self.cumsum_exp_delta
        ).update()

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

    @classmethod
    def from_nparam(
        cls,
        y: Array,
        locs: Array,
        nparam: int,
        knots_lo: float,
        knots_hi: float,
        kernel_class: type[
            tfk.AutoCompositeTensorPsdKernel
        ] = tfk.ExponentiatedQuadratic,
        extrap_transition_width: float = 0.3,
    ) -> Model:
        knots = ptm.kn(jnp.array([knots_lo, knots_hi]), order=3, n_params=nparam)
        return cls(
            y=y,
            knots=knots,
            locs=locs,
            kernel_class=kernel_class,
            extrap_transition_width=extrap_transition_width,
        )

    def build_graph(self):
        graph = lsl.GraphBuilder().add(self.response).build_model()
        return graph

    @property
    def eta_param_name(self) -> str:
        return self.eta.name

    @property
    def eta_hyperparam_names(self) -> list[str]:
        kernel_value = self.eta.dist_node.kwinputs["covariance_matrix"].inputs[0]
        amplitude_name = (
            kernel_value.kwinputs["amplitude"].var.value_node.inputs[0].var.name
        )
        length_scale_name = kernel_value.kwinputs["length_scale"].var.name
        return [amplitude_name, length_scale_name]

    @property
    def delta_param_name(self) -> str:
        return self.delta.value_node.kwinputs["latent_delta"].var.name

    @property
    def delta_hyperparam_names(self) -> list[str]:
        L = self.delta.value_node.kwinputs["L"]
        kernel_value = L.var.value_node.inputs[0].var.value_node
        amplitude_name = (
            kernel_value.kwinputs["amplitude"].var.value_node.inputs[0].var.name
        )
        length_scale_name = kernel_value.kwinputs["length_scale"].var.name
        return [amplitude_name, length_scale_name]

    @property
    def alpha_param_name(self) -> str:
        return self.alpha.value_node.inputs[0].var.name

    @property
    def alpha_hyperparam_names(self) -> list[str]:
        alpha_var = self.alpha.value_node.inputs[0].var
        kernel_value = alpha_var.dist_node.kwinputs["covariance_matrix"].inputs[0]
        amplitude_name = (
            kernel_value.kwinputs["amplitude"].var.value_node.inputs[0].var.name
        )
        length_scale_name = kernel_value.kwinputs["length_scale"].var.name
        return [amplitude_name, length_scale_name]

    @property
    def beta_param_name(self) -> str:
        return self.exp_beta.value_node.inputs[0].var.name

    @property
    def beta_hyperparam_names(self) -> list[str]:
        beta_var = self.exp_beta.value_node.inputs[0].var
        kernel_value = beta_var.dist_node.kwinputs["covariance_matrix"].inputs[0]
        amplitude_name = (
            kernel_value.kwinputs["amplitude"].var.value_node.inputs[0].var.name
        )
        length_scale_name = kernel_value.kwinputs["length_scale"].var.name
        return [amplitude_name, length_scale_name]


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
