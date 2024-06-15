from __future__ import annotations

import jax
import jax.numpy as jnp
import liesel.model as lsl
import liesel_ptm as ptm
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
import optax
from .node_ip import Kernel, Kernel2, Array
from .optim import optim_flat, OptimResult


class LocationParam(lsl.Var):
    def __init__(
        self,
        locs: Array,
        K: int,
        name: str = "loc",
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
        ).update()

        kernel_du = Kernel2(
            x1=locs[K:, :],
            x2=locs[:K, :],
            kernel_class=kernel_class,
            **kernel_args,
            name=f"kernel_latent_{name}_u",
        ).update()

        latent_param = lsl.param(
            jnp.zeros((K,)),
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name=f"latent_{name}",
        )

        constant = lsl.param(0.0, name=f"{name}_mean")

        def _compute_param(constant, latent_param, Kuu, Kdu):
            salt = jnp.diag(jnp.full(shape=(Kuu.shape[0],), fill_value=1e-6))
            Kuu = Kuu + salt
            L = jnp.linalg.cholesky(Kuu)
            Li = jnp.linalg.inv(L)

            u = L @ latent_param
            alpha = Kdu @ Li.T @ latent_param
            return constant + jnp.r_[u, alpha]

        super().__init__(
            lsl.Calc(_compute_param, constant, latent_param, kernel_uu, kernel_du), name=name
        )
        self.parameter_names = [latent_param.name]
        self.hyperparameter_names = [
            constant.name,
            amplitude_transformed.name,
            length_scale_transformed.name,
        ]


class ScaleParam(lsl.Var):
    def __init__(
        self,
        locs: Array,
        K: int,
        name: str = "scale",
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

        latent_param = lsl.param(
            jnp.zeros((K,)),
            distribution=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name=f"latent_{name}",
        )

        constant = lsl.param(0.0, name=f"{name}_mean")

        def _compute_param(constant, latent_param, Kuu, Kdu):
            salt = jnp.diag(jnp.full(shape=(Kuu.shape[0],), fill_value=1e-6))
            Kuu = Kuu + salt
            L = jnp.linalg.cholesky(Kuu)
            Li = jnp.linalg.inv(L)

            u = L @ latent_param
            alpha = Kdu @ Li.T @ latent_param
            return constant + jnp.exp(jnp.r_[u, alpha])

        super().__init__(
            lsl.Calc(_compute_param, constant, latent_param, kernel_uu, kernel_du), name=name
        )
        self.parameter_names = [latent_param.name]
        self.hyperparameter_names = [
            constant.name,
            amplitude_transformed.name,
            length_scale_transformed.name,
        ]


class Model:
    def __init__(
        self,
        y: Array,
        locs: Array,
        K: int,
        kernel_class: type[
            tfk.AutoCompositeTensorPsdKernel
        ] = tfk.ExponentiatedQuadratic,
    ) -> None:
        self.loc = LocationParam(locs, K=K, kernel_class=kernel_class).update()
        self.scale = ScaleParam(locs, K=K, kernel_class=kernel_class).update()

        self.response = lsl.obs(
            y, lsl.Dist(tfd.Normal, loc=self.loc, scale=self.scale), name="response"
        ).update()
        """Response variable."""

        self.graph = None


    def build_graph(self):
        self.graph = lsl.GraphBuilder().add(self.response).build_model()
        return self.graph
    
    def param(self):
        param = self.loc.parameter_names + self.scale.parameter_names
        return param
    
    def hyperparam(self):
        hyperparam = self.loc.hyperparameter_names + self.scale.hyperparameter_names
        return hyperparam

    def fit(
        self,
        graph: lsl.Model | None = None,
        graph_validation: lsl.Model | None = None,
        optimizer: optax.GradientTransformation | None = None,
        stopper: ptm.Stopper | None = None
    ) -> OptimResult:
        graph = self.graph if graph is None else graph

        result = optim_flat(
            graph,
            params=self.param() + self.hyperparam(),
            stopper=stopper,
            optimizer=optimizer,
            model_validation=graph_validation,
        )
        graph.state = result.model_state
        graph.update()
        return result

    def _normalization(self, y: Array) -> Array:
        """y has shape (Nobs, Nloc)"""
        return (y - self.loc.update().value) / self.scale.update().value

    def _normalization_log_deriv(self) -> Array:
        return -jnp.log(self.scale.update().value)

    def normalization_and_logdet(self, y: Array) -> tuple[Array, Array]:
        return self._normalization(y), self._normalization_log_deriv()

    def normalization_inverse(self, z: Array) -> Array:
        """y has shape (Nobs, Nloc)"""
        return z * self.scale.value + self.loc.value
