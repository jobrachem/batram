from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import liesel.model as lsl
import liesel_ptm as ptm
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from liesel.goose.optim import OptimResult, optim_flat
from liesel_ptm.dist import TransformationDist
from liesel_ptm.nodes import TransformationDistLogDeriv
from liesel_ptm.ptm_ls import NormalizationFn

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
        intercept: ParamPredictivePointProcessGP | ModelConst | None = None,
        slope: ParamPredictivePointProcessGP | ModelConst | None = None,
    ) -> None:
        self.knots = knots
        self.coef = coef

        if intercept is None:
            self.intercept = ModelConst(0.0, name="intercept")
        else:
            self.intercept = intercept

        if slope is None:
            self.slope = ModelConst(1.0, name="slope")
        else:
            self.slope = slope

        self._extrap_transition_width = 0.3
        self.bspline = ptm.ExtrapBSplineApprox(
            knots=knots, order=3, eps=self._extrap_transition_width
        )
        basis_dot_and_deriv_fn = self.bspline.get_extrap_basis_dot_and_deriv_fn(
            target_slope=1.0
        )

        self.response_value = lsl.obs(y, name="response_hidden_value")

        def trafo_and_deriv_fn(y, coef, intercept, slope):
            spline, spline_deriv = basis_dot_and_deriv_fn(y.T, coef.T)
            transformed = spline.T * slope + intercept
            transformation_deriv = spline_deriv.T * slope
            return transformed, transformation_deriv

        self.transformation_and_deriv = lsl.Var(
            lsl.Calc(
                trafo_and_deriv_fn,
                y=self.response_value,
                coef=self.coef,
                intercept=self.intercept,
                slope=self.slope,
            ),
            name="transformation_and_deriv",
        ).update()

        self.transformation = lsl.Var(
            lsl.Calc(lambda x: x[0], self.transformation_and_deriv),
            name="transformation",
        ).update()

        self.transformation_deriv = lsl.Var(
            lsl.Calc(lambda x: x[1], self.transformation_and_deriv),
            name="transformation_deriv",
        ).update()

        self.refdist = tfd.Normal(loc=0.0, scale=1.0)
        """y
        The reference distribution, currently fixed to the standard normal distribution.
        """

        response_dist = ptm.TransformationDist(
            self.transformation, self.transformation_deriv, refdist=self.refdist
        )
        self.response = lsl.obs(y, response_dist, name="response").update()
        """Response variable."""

        self.graph = lsl.GraphBuilder().add(self.response).build_model()

    def param_names(self) -> list[str]:
        names: list[str] = []
        names += self.coef.parameter_names
        names += self.intercept.parameter_names
        names += self.slope.parameter_names
        return list(set(names))

    def hyperparam_names(self) -> list[str]:
        names: list[str] = []
        names += self.coef.hyperparameter_names
        names += self.intercept.hyperparameter_names
        names += self.slope.hyperparameter_names
        return list(set(names))

    def copy_for(
        self, y: Array, sample_locs: lsl.Var | lsl.Node | None = None
    ) -> TransformationModel:
        coef = self.coef.copy_for(sample_locs)
        intercept = self.intercept.copy_for(sample_locs)
        slope = self.slope.copy_for(sample_locs)

        model = TransformationModel(
            y=y, knots=self.knots, coef=coef, intercept=intercept, slope=slope
        )

        return model

    def fit_loc_batched(
        self,
        y: Array,
        locs: Array,
        model_validation: TransformationModel,
        loc_batch_size: int | None = None,
        optimizer: optax.GradientTransformation | None = None,
        stopper: ptm.Stopper | None = None,
    ) -> OptimResult:
        locs_name = self.coef.latent_coef.sample_locs.name

        model_batched = self.copy_for(
            y=self.response.value[:, :loc_batch_size],
            sample_locs=lsl.Var(locs[:loc_batch_size, ...], name=locs_name),
        )
        graph_batched = model_batched.graph

        model_validation_batched = model_validation.copy_for(
            y=model_validation.response.value[:, :loc_batch_size],
            sample_locs=lsl.Var(locs[:loc_batch_size, ...], name=locs_name),
        )

        graph_validation_batched = model_validation_batched.graph

        result = optim_loc_batched(
            graph_batched,
            params=self.param_names() + self.hyperparam_names(),
            stopper=stopper,
            optimizer=optimizer,
            response=lsl.Var(y, name="response"),
            locs=lsl.Var(locs, name=locs_name),
            loc_batch_size=loc_batch_size,
            model_validation=graph_validation_batched,
        )

        graph_batched.state = result.model_state
        graph_batched.update()

        self.coef.update_from(model_batched.coef)
        self.intercept.update_from(model_batched.intercept)
        self.slope.update_from(model_batched.slope)

        return result

    def transformation_and_logdet(
        self, y: Array, locs: Array | None = None
    ) -> tuple[Array, Array]:
        if locs is None:
            locs = self.coef.latent_coef.sample_locs.value

        n_loc = locs.shape[0]
        if n_loc != y.shape[1]:
            raise ValueError(
                f"Number of available locations ({n_loc}) != Number of locations in y"
                f" ({y.shape[1]})"
            )

        n_loc_model = self.response.value.shape[1]

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
        response_value = graph.vars[self.response_value.name]
        coef = graph.vars[self.coef.name]
        transformation = graph.vars[self.transformation.name]
        transformation_deriv = graph.vars[self.transformation_deriv.name]

        def one_batch(y, locs):
            response_value.value = y
            coef.latent_coef.sample_locs.value = locs
            graph.update()
            z = transformation.value
            logdet = jnp.log(transformation_deriv.value)
            return z, logdet

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
        z_i = model_last_batch.transformation.value
        z_logdet_i = jnp.log(model_last_batch.transformation_deriv.value)

        z = z.at[:, last_batch_indices].set(z_i)
        z_logdet = z_logdet.at[:, last_batch_indices].set(z_logdet_i)

        return z, z_logdet

    def transformation_inverse(self, z: Array, locs: Array | None = None) -> Array:
        """
        Warning: Does not take intercept or slope into account!
        """
        if locs is None:
            locs = self.coef.latent_coef.sample_locs.value

        n_loc = locs.shape[0]
        if n_loc != z.shape[1]:
            raise ValueError(
                f"Number of available locations ({n_loc}) != Number of locations in z"
                f" ({z.shape[1]})"
            )

        n_loc_model = self.response.value.shape[1]

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
                coef=coef.update().value.T,
            )

            y = dist.inverse_transformation(z)

            return y

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
            coef=model_last_batch.coef.update().value.T,
        )
        y_i = dist.inverse_transformation(z_last_batch)
        y = y.at[:, last_batch_indices].set(y_i)

        return y


class ChainedModel(Model):
    def __init__(
        self,
        y: Array,
        tfp_dist_cls: type[tfd.Distribution],
        knots: Array,
        coef: OnionCoefPredictivePointProcessGP,
        **params,
    ) -> None:
        self.param = params

        def apriori_transformation_and_logdet_fn(y, **params):
            dist = tfp_dist_cls(**params)
            u = dist.cdf(y)

            normal = tfd.Normal(loc=0.0, scale=1.0)
            z = normal.quantile(u)

            logdet = dist.log_prob(y) - normal.log_prob(z)
            return z, logdet

        self.raw_response_value = lsl.obs(y, name="raw_response_value")

        self.apriori_transformation_and_logdet = lsl.Var(
            lsl.Calc(
                apriori_transformation_and_logdet_fn, self.raw_response_value, **params
            ),
            name="normalization_and_logdet",
        )

        self.knots = knots
        self.coef = coef
        self._extrap_transition_width = 0.3
        self.bspline = ptm.ExtrapBSplineApprox(
            knots=knots, order=3, eps=self._extrap_transition_width
        )
        basis_dot_and_deriv_fn = self.bspline.get_extrap_basis_dot_and_deriv_fn(
            target_slope=1.0
        )

        self.response_value = lsl.Var(
            lsl.Calc(lambda x: x[0], self.apriori_transformation_and_logdet),
            name="response_hidden_value",
        )

        self.transformation_and_deriv = lsl.Var(
            lsl.Calc(
                lambda y, c: basis_dot_and_deriv_fn(y.T, c.T),
                self.response_value,
                self.coef,
            ),
            name="transformation_and_deriv",
        ).update()

        self.transformation = lsl.Var(
            lsl.Calc(lambda x: x[0].T, self.transformation_and_deriv),
            name="transformation",
        ).update()

        def log_deriv(apriori_transformation_and_logdet, transformation_and_deriv):
            return (
                apriori_transformation_and_logdet[1]
                + jnp.log(transformation_and_deriv[1]).T
            )

        self.log_det = lsl.Var(
            lsl.Calc(
                log_deriv,
                self.apriori_transformation_and_logdet,
                self.transformation_and_deriv,
            ),
            name="log_det",
        ).update()

        self.refdist = tfd.Normal(loc=0.0, scale=1.0)
        """
        The reference distribution, currently fixed to the standard normal distribution.
        """

        response_dist = TransformationDistLogDeriv(
            self.transformation, self.log_det, refdist=self.refdist
        )
        self.response = lsl.obs(y, response_dist, name="response").update()
        """Response variable."""

        self.graph = lsl.GraphBuilder().add(self.response).build_model()

    def param_names(self) -> list[str]:
        param_names: list[str] = []
        for param in self.params.values():
            param_names += param.parameter_names

        return param_names + self.coef.parameter_names

    def hyperparam_names(self) -> list[str]:
        hyper_param_names: list[str] = []
        for param in self.params.values():
            hyper_param_names += param.hyperparameter_names
        return hyper_param_names + self.coef.hyper_parameter_names

    def transformation_and_logdet(self, y: Array) -> tuple[Array, Array]:
        _, vars_ = self.graph.copy_nodes_and_vars()
        graph_copy = lsl.GraphBuilder().add(vars_[self.response.name]).build_model()
        graph_copy.vars[self.raw_response_value.name].value = y
        graph_copy.update()
        z = graph_copy.vars[self.transformation.name].value
        logdet = graph_copy.vars[self.log_det.name].value
        return z, logdet

    def transformation_inverse(self, z: Array) -> Array:
        hfn = NormalizationFn(
            knots=self.knots,
            order=3,
            transition_width=self._extrap_transition_width,
        )

        y_after_trafo = hfn.inverse_newton(
            z=z.T,
            coef=self.coef.update().value,
            norm_mean=jnp.zeros(1),
            norm_sd=jnp.ones(1),
        ).T

        normal = tfd.Normal(loc=0.0, scale=1.0)
        u = normal.cdf(y_after_trafo)

        param_values = {name: node.update().value for name, node in self.params.items()}
        dist = self.tfp_dist_cls(**param_values)

        y = dist.quantile(u)

        return y
