from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import liesel.model as lsl
import liesel_ptm as ptm
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from liesel.goose.optim import OptimResult, optim_flat
from liesel_ptm.bsplines import ExtrapBSplineApprox
from liesel_ptm.dist import LocScaleTransformationDist, TransformationDist

from .node import (
    ModelConst,
    ModelOnionCoef,
    ModelVar,
    OnionCoefPredictivePointProcessGP,
    ParamPredictivePointProcessGP,
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
        parametric_distribution: type[tfd.Distribution] | None = None,
        to_float32: bool = True,
        **parametric_distribution_kwargs: (
            ModelConst | ModelVar | ParamPredictivePointProcessGP
        ),
    ) -> None:
        self.knots = knots
        self.coef = coef
        self.parametric_distribution_kwargs = parametric_distribution_kwargs

        bspline = ExtrapBSplineApprox(knots=knots, order=3)
        self.fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)
        self.parametric_distribution = parametric_distribution

        self.dist_class = partial(
            TransformationDist,
            knots=knots,
            basis_dot_and_deriv_fn=self.fn,
            parametric_distribution=self.parametric_distribution,
        )

        response_dist = lsl.Dist(
            self.dist_class,
            coef=coef,
            **parametric_distribution_kwargs,
        )
        self.response = lsl.obs(y.T, response_dist, name="response").update()
        """Response variable."""

        self._to_float32 = to_float32
        gb = lsl.GraphBuilder(to_float32=to_float32).add(self.response)
        self.graph = gb.build_model()

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
        parametric_distributionargs = {
            name: var_.copy_for(sample_locs)
            for name, var_ in self.parametric_distribution_kwargs.items()
        }

        model = TransformationModel(
            y=y,
            knots=self.knots,
            coef=coef,
            parametric_distribution=self.parametric_distribution,
            to_float32=self._to_float32,
            **parametric_distributionargs,
        )

        return model

    def fit_loc_batched(
        self,
        train: Array,
        validation: Array,
        locs: lsl.Var,
        optimizer: optax.GradientTransformation | None = None,
        stopper: ptm.Stopper | None = None,
    ) -> OptimResult:
        result = optim_loc_batched(
            model=self.graph,
            params=self.coef.parameter_names + self.coef.hyperparameter_names,
            stopper=stopper,
            optimizer=optimizer,
            response_train=lsl.Var(jnp.asarray(train.T), name="response"),
            response_validation=lsl.Var(jnp.asarray(validation.T), name="response"),
            locs=locs,
            loc_batch_size=self.response.value.shape[0],
        )

        self.graph.state = result.model_state
        self.graph.update()

        return result

    def fit_parametric_distributionloc_batched(
        self,
        train: Array,
        validation: Array,
        locs: lsl.Var,
        optimizer: optax.GradientTransformation | None = None,
        stopper: ptm.Stopper | None = None,
    ) -> OptimResult:
        params: list[str] = []
        hyper_params: list[str] = []
        for var_ in self.parametric_distribution_kwargs.values():
            params += var_.parameter_names
            hyper_params += var_.hyperparameter_names

        result = optim_loc_batched(
            model=self.graph,
            params=params + hyper_params,
            stopper=stopper,
            optimizer=optimizer,
            response_train=lsl.Var(jnp.asarray(train.T), name="response"),
            response_validation=lsl.Var(jnp.asarray(validation.T), name="response"),
            locs=locs,
            loc_batch_size=self.response.value.shape[0],
        )

        self.graph.state = result.model_state
        self.graph.update()

        return result

    def transformation_and_logdet_parametric(
        self, y: Array, locs: Array | None = None
    ) -> tuple[Array, Array]:
        return self._transformation_and_logdet(
            y, locs, which="transformation_and_logdet_parametric"
        )

    def transformation_and_logdet_spline(
        self, y: Array, locs: Array | None = None
    ) -> tuple[Array, Array]:
        return self._transformation_and_logdet(
            y, locs, which="transformation_and_logdet_spline"
        )

    def transformation_and_logdet(
        self, y: Array, locs: Array | None = None
    ) -> tuple[Array, Array]:
        return self._transformation_and_logdet(
            y, locs, which="transformation_and_logdet"
        )

    def _transformation_and_logdet(
        self,
        y: Array,
        locs: Array | None = None,
        which: str = "transformation_and_logdet",
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
        graph = (
            lsl.GraphBuilder(to_float32=self._to_float32)
            .add(_vars["response"])
            .build_model()
        )
        coef = graph.vars[self.coef.name]
        parametric_distributionargs = {
            name: graph.vars[var_.name]
            for name, var_ in self.parametric_distribution_kwargs.items()
        }

        def one_batch(y, locs):
            coef.latent_coef.sample_locs.value = locs

            parametric_distribution_values = dict()
            for name, var_ in parametric_distributionargs.items():
                var_.set_locs(locs)
                parametric_distribution_values[name] = var_.value

            dist = self.dist_class(coef=coef.value, **parametric_distribution_values)
            transformation_and_logdet_fn = getattr(dist, which)
            z, logdet = transformation_and_logdet_fn(y.T)
            return z.T, logdet.T

        z = jnp.empty_like(y)
        z_logdet = jnp.empty_like(y)
        init_val = (y, locs, batch_indices, z, z_logdet)
        one_batch(y[:, batch_indices[0]], locs[batch_indices[0], ...])

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
        parametric_distributionargs_last_batch = {
            name: model_last_batch.graph.vars[var_.name].value
            for name, var_ in self.parametric_distribution_kwargs.items()
        }

        dist = self.dist_class(
            coef=model_last_batch.coef.value, **parametric_distributionargs_last_batch
        )
        transformation_and_logdet_fn = getattr(dist, which)
        z_i, z_logdet_i = transformation_and_logdet_fn(y_last_batch.T)

        z = z.at[:, last_batch_indices].set(z_i.T)
        z_logdet = z_logdet.at[:, last_batch_indices].set(z_logdet_i.T)

        return z, z_logdet

    def transformation_inverse_parametric(
        self, z: Array, locs: Array | None = None
    ) -> Array:
        return self._transformation_inverse(
            z, locs, which="inverse_transformation_parametric"
        )

    def transformation_inverse_spline(
        self, z: Array, locs: Array | None = None
    ) -> Array:
        return self._transformation_inverse(
            z, locs, which="inverse_transformation_spline"
        )

    def transformation_inverse(self, z: Array, locs: Array | None = None) -> Array:
        return self._transformation_inverse(z, locs, which="inverse_transformation")

    def _transformation_inverse(
        self, z: Array, locs: Array | None = None, which: str = "inverse_transformation"
    ) -> Array:
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
        graph = (
            lsl.GraphBuilder(to_float32=self._to_float32)
            .add(_vars["response"])
            .build_model()
        )
        coef = graph.vars[self.coef.name]
        parametric_distributionargs = {
            name: graph.vars[var_.name]
            for name, var_ in self.parametric_distribution_kwargs.items()
        }

        def one_batch(z, locs):
            coef.latent_coef.sample_locs.value = locs

            parametric_distribution_values = dict()
            for name, var_ in parametric_distributionargs.items():
                var_.set_locs(locs)
                parametric_distribution_values[name] = var_.value

            dist = self.dist_class(
                coef=coef.update().value, **parametric_distribution_values
            )

            inverse_transformation_fn = getattr(dist, which)

            y = inverse_transformation_fn(z.T)

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
        parametric_distributionargs_last_batch = {
            name: model_last_batch.graph.vars[var_.name].value
            for name, var_ in self.parametric_distribution_kwargs.items()
        }

        dist = self.dist_class(
            coef=model_last_batch.coef.value, **parametric_distributionargs_last_batch
        )
        inverse_transformation_fn = getattr(dist, which)
        y_i = inverse_transformation_fn(z_last_batch.T)
        y = y.at[:, last_batch_indices].set(y_i.T)

        return y


class LocScaleTransformationModel(TransformationModel):
    """
    Dedicated :class:`.TransformationModel` for location-scale
    """

    def __init__(
        self,
        y: Array,
        knots: Array,
        coef: OnionCoefPredictivePointProcessGP | ModelOnionCoef,
        loc: (ModelConst | ModelVar | ParamPredictivePointProcessGP),
        scale: (ModelConst | ModelVar | ParamPredictivePointProcessGP),
        to_float32: bool = True,
    ) -> None:
        self.knots = knots
        self.coef = coef
        self.parametric_distribution_kwargs = {"loc": loc, "scale": scale}
        self.loc = loc
        self.scale = scale

        bspline = ExtrapBSplineApprox(knots=knots, order=3)
        self.fn = bspline.get_extrap_basis_dot_and_deriv_fn(target_slope=1.0)
        self.parametric_distribution = tfd.Normal

        self.dist_class = partial(
            LocScaleTransformationDist,
            knots=knots,
            basis_dot_and_deriv_fn=self.fn,
        )

        response_dist = lsl.Dist(
            self.dist_class,
            coef=coef,
            **self.parametric_distribution_kwargs,
        )
        self.response = lsl.obs(y.T, response_dist, name="response").update()
        """Response variable."""

        self._to_float32 = to_float32
        self.graph = (
            lsl.GraphBuilder(to_float32=self._to_float32)
            .add(self.response)
            .build_model()
        )


class GEVTransformationModel(TransformationModel):
    def copy_for(
        self, y: Any, sample_locs: lsl.Var | lsl.Node | None = None
    ) -> TransformationModel:
        coef = self.coef.copy_for(sample_locs)

        location = self.parametric_distribution_kwargs["loc"]
        location, scale, concentration = location.copy_for(sample_locs=sample_locs)

        model = TransformationModel(
            y=y,
            knots=self.knots,
            coef=coef,
            parametric_distribution=self.parametric_distribution,
            to_float32=self._to_float32,
            loc=location,
            scale=scale,
            concentration=concentration,
        )

        return model
