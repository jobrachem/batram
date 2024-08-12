from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import liesel.model as lsl
import liesel_ptm as ptm
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from liesel.goose.optim import OptimResult, optim_flat
from liesel_ptm.bsplines import ExtrapBSplineApprox
from liesel_ptm.dist import TransformationDist

from .node import (
    ModelConst,
    ModelOnionCoef,
    ModelVar,
    OnionCoefPredictivePointProcessGP,
)
from .optim import optim_loc_batched

Array = Any


class Model:
    def __init__(
        self,
        y: Array,
        tfp_dist_cls: type[tfd.Distribution],
        **params: ModelConst | ModelVar,
    ) -> None:
        self.params = params
        self.tfp_dist_cls = tfp_dist_cls

        self.response = lsl.obs(
            y,
            lsl.Dist(
                self.tfp_dist_cls,
                **params,
            ),
            name="response",
        ).update()
        """Response variable."""

        self.graph = lsl.GraphBuilder().add(self.response).build_model()

    def param_names(self) -> list[str]:
        param_names: list[str] = []
        for param in self.params.values():
            param_names += param.parameter_names
        return list(set(param_names))

    def hyperparam_names(self) -> list[str]:
        hyper_param_names = []
        for param in self.params.values():
            hyper_param_names += param.hyperparameter_names
        return list(set(hyper_param_names))

    def fit(
        self,
        graph: lsl.Model | None = None,
        graph_validation: lsl.Model | None = None,
        optimizer: optax.GradientTransformation | None = None,
        stopper: ptm.Stopper | None = None,
    ) -> OptimResult:
        if graph is None:
            graph = self.graph

        result = optim_flat(
            graph,
            params=self.param_names() + self.hyperparam_names(),
            stopper=stopper,
            optimizer=optimizer,
            model_validation=graph_validation,
        )
        graph.state = result.model_state
        graph.update()
        return result

    def transformation_and_logdet(self, y: Array) -> tuple[Array, Array]:
        param_values = {name: node.update().value for name, node in self.params.items()}
        dist = self.tfp_dist_cls(**param_values)
        u = dist.cdf(y)

        normal = tfd.Normal(loc=0.0, scale=1.0)
        z = normal.quantile(u)
        logdet = dist.log_prob(y) - normal.log_prob(z)

        return z, logdet

    def transformation_inverse(self, z: Array) -> Array:
        normal = tfd.Normal(loc=0.0, scale=1.0)
        u = normal.cdf(z)

        param_values = {name: node.update().value for name, node in self.params.items()}
        dist = self.tfp_dist_cls(**param_values)

        y = dist.quantile(u)

        return y

    def normalization_and_logdet(self, y: Array) -> tuple[Array, Array]:
        return self.transformation_and_logdet(y)

    def normalization_inverse(self, z: Array) -> Array:
        return self.transformation_inverse(z)


class TransformationModel(Model):
    def __init__(
        self,
        y: Array,
        knots: Array,
        coef: OnionCoefPredictivePointProcessGP | ModelOnionCoef,
    ) -> None:
        self.knots = knots
        self.coef = coef

        bspline = ExtrapBSplineApprox(knots=knots, order=3)
        self.fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)

        response_dist = lsl.Dist(
            TransformationDist, knots=knots, coef=coef, basis_dot_and_deriv_fn=self.fn
        )
        self.response = lsl.obs(y.T, response_dist, name="response").update()
        """Response variable."""

        self.graph = lsl.GraphBuilder().add(self.response).build_model()

    def param_names(self) -> list[str]:
        names: list[str] = []
        names += self.coef.parameter_names
        return list(set(names))

    def hyperparam_names(self) -> list[str]:
        names: list[str] = []
        names += self.coef.hyperparameter_names
        return list(set(names))

    def copy_for(
        self, y: Array, sample_locs: lsl.Var | lsl.Node | None = None
    ) -> TransformationModel:
        coef = self.coef.copy_for(sample_locs)

        model = TransformationModel(y=y, knots=self.knots, coef=coef)

        return model

    def fit_loc_batched(
        self,
        train: Array,
        validation: Array,
        locs: Array,
        optimizer: optax.GradientTransformation | None = None,
        stopper: ptm.Stopper | None = None,
    ) -> OptimResult:
        locs_name = self.coef.latent_coef.sample_locs.name

        result = optim_loc_batched(
            model=self.graph,
            params=self.param_names() + self.hyperparam_names(),
            stopper=stopper,
            optimizer=optimizer,
            response_train=lsl.Var(jnp.asarray(train.T), name="response"),
            response_validation=lsl.Var(jnp.asarray(validation.T), name="response"),
            locs=lsl.Var(jnp.asarray(locs), name=locs_name),
            loc_batch_size=self.response.value.shape[0],
        )

        self.graph.state = result.model_state
        self.graph.update()

        return result

    def transformation_and_logdet(
        self, y: Array, locs: Array | None = None
    ) -> tuple[Array, Array]:
        y = jnp.asarray(y)
        if locs is None:
            locs = self.coef.latent_coef.sample_locs.value
        else:
            locs = jnp.asarray(locs)

        n_loc = locs.shape[0]
        n_loc_model = self.response.value.shape[0]

        def _generate_batch_indices(n: int, batch_size: int) -> Array:
            n_full_batches = n // batch_size
            indices = jnp.arange(n)
            indices_subset = indices[0 : n_full_batches * batch_size]
            list_of_batch_indices = jnp.array_split(indices_subset, n_full_batches)
            return jnp.asarray(list_of_batch_indices)

        batch_indices = _generate_batch_indices(n_loc, batch_size=n_loc_model)
        last_batch_indices = jnp.arange(batch_indices[-1, -1] + 1, y.shape[1])

        _, _vars = self.graph.copy_nodes_and_vars()
        graph = lsl.GraphBuilder().add(_vars["response"]).build_model()
        coef = graph.vars[self.coef.name]

        def one_batch(y, locs):
            coef.latent_coef.sample_locs.value = locs
            dist = TransformationDist(
                self.knots, coef.value, basis_dot_and_deriv_fn=self.fn
            )
            z, logdet = dist.transformation_and_logdet(y.T)
            return z.T, logdet.T

        z = jnp.empty_like(y)
        z_logdet = jnp.empty_like(y)
        init_val = (y, locs, batch_indices, z, z_logdet)

        def body_fun(i, val):
            y, locs, batch_indices, z, z_logdet = val

            idx = batch_indices[i]

            z_i, z_logdet_i = one_batch(y[:, idx], locs[idx, ...])
            z = z.at[:, batch_indices[i]].set(z_i)
            z_logdet = z_logdet.at[:, batch_indices[i]].set(z_logdet_i)

            return (y, locs, batch_indices, z, z_logdet)

        _, _, _, z, z_logdet = jax.lax.fori_loop(
            lower=0, upper=batch_indices.shape[0], body_fun=body_fun, init_val=init_val
        )

        y_last_batch = y[:, last_batch_indices]
        locs_last_batch = locs[last_batch_indices, ...]
        model_last_batch = self.copy_for(
            y=y_last_batch, sample_locs=lsl.Var(locs_last_batch, name="locs")
        )
        dist = TransformationDist(
            self.knots, model_last_batch.coef.value, basis_dot_and_deriv_fn=self.fn
        )
        z_i, z_logdet_i = dist.transformation_and_logdet(y_last_batch.T)

        z = z.at[:, last_batch_indices].set(z_i.T)
        z_logdet = z_logdet.at[:, last_batch_indices].set(z_logdet_i.T)

        return z, z_logdet

    def transformation_inverse(self, z: Array, locs: Array | None = None) -> Array:
        """
        Warning: Does not take intercept or slope into account!
        """
        z = jnp.asarray(z)
        if locs is None:
            locs = self.coef.latent_coef.sample_locs.value
        else:
            locs = jnp.asarray(locs)

        n_loc = locs.shape[0]
        n_loc_model = self.response.value.shape[0]

        def _generate_batch_indices(n: int, batch_size: int) -> Array:
            n_full_batches = n // batch_size
            indices = jnp.arange(n)
            indices_subset = indices[0 : n_full_batches * batch_size]
            list_of_batch_indices = jnp.array_split(indices_subset, n_full_batches)
            return jnp.asarray(list_of_batch_indices)

        batch_indices = _generate_batch_indices(n_loc, batch_size=n_loc_model)
        last_batch_indices = jnp.arange(batch_indices[-1, -1] + 1, z.shape[1])

        _, _vars = self.graph.copy_nodes_and_vars()
        graph = lsl.GraphBuilder().add(_vars["response"]).build_model()
        coef = graph.vars[self.coef.name]

        def one_batch(z, locs):
            coef.latent_coef.sample_locs.value = locs

            dist = TransformationDist(
                knots=self.knots,
                coef=coef.update().value,
                basis_dot_and_deriv_fn=self.fn,
            )

            y = dist.inverse_transformation(z.T)

            return y.T

        y = jnp.empty_like(z)
        init_val = (z, locs, batch_indices, y)

        def body_fun(i, val):
            z, locs, batch_indices, y = val
            idx = batch_indices[i]
            y_i = one_batch(z[:, idx], locs[idx, ...])
            y = y.at[:, batch_indices[i]].set(y_i)
            return (z, locs, batch_indices, y)

        _, _, _, y = jax.lax.fori_loop(
            lower=0, upper=batch_indices.shape[0], body_fun=body_fun, init_val=init_val
        )

        z_last_batch = z[:, last_batch_indices]
        locs_last_batch = locs[last_batch_indices, ...]
        model_last_batch = self.copy_for(
            y=z_last_batch, sample_locs=lsl.Var(locs_last_batch, name="locs")
        )
        dist = TransformationDist(
            knots=self.knots,
            coef=model_last_batch.coef.update().value,
            basis_dot_and_deriv_fn=self.fn,
        )
        y_i = dist.inverse_transformation(z_last_batch.T)
        y = y.at[:, last_batch_indices].set(y_i.T)

        return y
