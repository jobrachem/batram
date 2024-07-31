from __future__ import annotations

import liesel.model as lsl
import liesel_ptm as ptm
import tensorflow_probability.substrates.jax.distributions as tfd
import optax
from .node_ip import Array
from liesel.goose.optim import optim_flat, OptimResult
from liesel_ptm.ptm_ls import NormalizationFn
import jax.numpy as jnp
from .ppnode import (
    OnionCoefPredictivePointProcessGP,
    ParamPredictivePointProcessGP,
    ModelConst,
)
from liesel_ptm.nodes import TransformationDistLogDeriv


class Model:
    def __init__(
        self, y: Array, tfp_dist_cls: type[tfd.Distribution], **params
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

        self.graph = None

    def build_graph(self):
        self.graph = lsl.GraphBuilder().add(self.response).build_model()
        return self.graph

    def param_names(self) -> list[str]:
        param_names = []
        for param in self.params:
            param_names += param.parameter_names
        return list(set(param_names))

    def hyperparam_names(self) -> list[str]:
        hyper_param_names = []
        for param in self.params:
            hyper_param_names += param.hyperparameter_names
        return list(set(hyper_param_names))

    def fit(
        self,
        graph: lsl.Model | None = None,
        graph_validation: lsl.Model | None = None,
        optimizer: optax.GradientTransformation | None = None,
        stopper: ptm.Stopper | None = None,
    ) -> OptimResult:
        graph = self.graph if graph is None else graph

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
        coef: OnionCoefPredictivePointProcessGP,
        intercept: ParamPredictivePointProcessGP | ModelConst | None = None,
        slope: ParamPredictivePointProcessGP | ModelConst | None = None,
    ) -> None:
        self.knots = knots
        self.coef = coef

        self.intercept = intercept
        if intercept is None:
            self.intercept = ModelConst(0.0, name="intercept")

        self.slope = slope
        if slope is None:
            self.slope = ModelConst(1.0, name="slope")

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
            transformed = spline * slope + intercept
            transformation_deriv = spline_deriv * slope
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
            lsl.Calc(lambda x: x[0].T, self.transformation_and_deriv),
            name="transformation",
        ).update()

        self.transformation_deriv = lsl.Var(
            lsl.Calc(lambda x: x[1].T, self.transformation_and_deriv),
            name="transformation_deriv",
        ).update()

        self.refdist = tfd.Normal(loc=0.0, scale=1.0)
        """
        The reference distribution, currently fixed to the standard normal distribution.
        """

        response_dist = ptm.TransformationDist(
            self.transformation, self.transformation_deriv, refdist=self.refdist
        )
        self.response = lsl.obs(y, response_dist, name="response").update()
        """Response variable."""

        self.graph = None

    def param_names(self) -> list[str]:
        return self.coef.parameter_names

    def hyperparam_names(self) -> list[str]:
        return self.coef.hyperparameter_names

    def transformation_and_logdet(self, y: Array) -> tuple[Array, Array]:
        _, vars_ = self.graph.copy_nodes_and_vars()
        graph_copy = lsl.GraphBuilder().add(vars_[self.response.name]).build_model()
        graph_copy.vars[self.response_value.name].value = y
        graph_copy.update()
        z = graph_copy.vars[self.transformation.name].value
        logdet = jnp.log(graph_copy.vars[self.transformation_deriv.name].value)
        return z, logdet

    def transformation_inverse(self, z: Array) -> Array:
        hfn = NormalizationFn(
            knots=self.knots,
            order=3,
            transition_width=self._extrap_transition_width,
        )

        y = hfn.inverse_newton(
            z=z.T,
            coef=self.coef.update().value.T,
            norm_mean=jnp.zeros(1),
            norm_sd=jnp.ones(1),
        ).T

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

        def normalization_and_logdet_fn(y, **params):
            dist = tfp_dist_cls(**params)
            u = dist.cdf(y)

            normal = tfd.Normal(loc=0.0, scale=1.0)
            z = normal.quantile(u)

            logdet = dist.log_prob(y) - normal.log_prob(z)
            return z, logdet

        self.raw_response_value = lsl.obs(y, name="raw_response_value")

        self.normalization_and_logdet = lsl.Var(
            lsl.Calc(normalization_and_logdet_fn, self.raw_response_value, **params),
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
            lsl.Calc(lambda x: x[0], self.normalization_and_logdet),
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

        def log_deriv(normalization_and_logdet, transformation_and_deriv):
            return normalization_and_logdet[1] + jnp.log(transformation_and_deriv[1]).T

        self.log_det = lsl.Var(
            lsl.Calc(
                log_deriv, self.normalization_and_logdet, self.transformation_and_deriv
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

        self.graph = None

    def param_names(self) -> list[str]:
        param_names = []
        for param in self.params:
            param_names += param.parameter_names

        return param_names + self.coef.parameter_names

    def hyperparam_names(self) -> list[str]:
        hyper_param_names = []
        for param in self.params:
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
