import jax
import jax.numpy as jnp
import jax.random as jrd
import liesel.model as lsl
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk

from batram.tmspat_jax.model import ChainedModel, Model, TransformationModel
from batram.tmspat_jax.node import (
    ModelOnionCoef,
    OnionCoefPredictivePointProcessGP,
    OnionKnots,
    ParamPredictivePointProcessGP,
)

key = jrd.PRNGKey(42)


class MockParam(lsl.Var):
    parameter = True
    parameter_names = []
    hyperparameter_names = []


class TestModel:
    def test_init(self):
        y = jrd.normal(key, shape=(20,))

        loc = MockParam(0.0, name="loc")
        scale = MockParam(1.0, name="scale")

        Model(y=y, tfp_dist_cls=tfd.Normal, loc=loc, scale=scale)
        assert True

    def test_init_locs(self):
        y = jrd.normal(key, shape=(20, 50))

        loc = MockParam(jnp.zeros(50), name="loc")
        scale = MockParam(jnp.ones(50), name="scale")

        Model(y=y, tfp_dist_cls=tfd.Normal, loc=loc, scale=scale)
        assert True

    @pytest.mark.parametrize(
        "loc,scale", [(0.0, 1.0), (0.0, 2.0), (1.0, 1.0), (1.0, 2.0)]
    )
    def test_transformation_and_logdet_normal(self, loc, scale):
        y = jrd.normal(key, shape=(20,))

        model = Model(
            y=y, tfp_dist_cls=tfd.Normal, loc=MockParam(loc), scale=MockParam(scale)
        )

        z, logdet = model.transformation_and_logdet(y)

        assert jnp.allclose((y - loc) / scale, z)
        assert jnp.allclose(logdet, -jnp.log(scale), atol=1e-6)

    @pytest.mark.parametrize(
        "loc,scale", [(0.0, 1.0), (0.0, 2.0), (1.0, 1.0), (1.0, 2.0)]
    )
    def test_transformation_and_logdet_normal_locs(self, loc, scale):
        y = jrd.normal(key, shape=(20, 50))

        loc_param = MockParam(jnp.full(shape=(50,), fill_value=loc))
        scale_param = MockParam(jnp.full(shape=(50,), fill_value=scale))
        model = Model(y=y, tfp_dist_cls=tfd.Normal, loc=loc_param, scale=scale_param)

        z, logdet = model.transformation_and_logdet(y)

        assert z.shape == y.shape

        assert jnp.allclose((y - loc_param.value) / scale_param.value, z, atol=1e-5)
        assert jnp.allclose(logdet, -jnp.log(scale_param.value), atol=1e-4)

    @pytest.mark.parametrize(
        "loc,scale", [(0.0, 1.0), (0.0, 2.0), (1.0, 1.0), (1.0, 2.0)]
    )
    def test_transformation_inverse_normal(self, loc, scale):
        y = jrd.normal(key, shape=(20,))

        model = Model(
            y=y, tfp_dist_cls=tfd.Normal, loc=MockParam(loc), scale=MockParam(scale)
        )

        z = jnp.linspace(-2.0, 2.0, 30)

        ynew = model.transformation_inverse(z)

        assert jnp.allclose(z * scale + loc, ynew)

    @pytest.mark.parametrize(
        "loc,scale", [(0.0, 1.0), (0.0, 2.0), (1.0, 1.0), (1.0, 2.0)]
    )
    def test_transformation_inverse_normal_locs(self, loc, scale):
        y = jrd.normal(key, shape=(20, 50))

        loc_param = MockParam(jnp.full(shape=(50,), fill_value=loc))
        scale_param = MockParam(jnp.full(shape=(50,), fill_value=scale))
        model = Model(y=y, tfp_dist_cls=tfd.Normal, loc=loc_param, scale=scale_param)

        z = jnp.linspace(-2.0, 2.0, 50)
        z = jnp.c_[z, z].T
        ynew = model.transformation_inverse(z)

        assert ynew.shape == (2, 50)

        assert jnp.allclose(ynew, z * scale_param.value + loc_param.value, atol=1e-5)


class TestTransformationModel:
    def test_init(self):
        y = jrd.normal(key, shape=(20, 50))
        locs = jrd.uniform(key, shape=(y.shape[1], 2))

        knots = OnionKnots(-3.0, 3.0, nparam=12)
        coef = OnionCoefPredictivePointProcessGP.new_from_locs(
            knots,
            inducing_locs=lsl.Var(locs[:5, :]),
            sample_locs=lsl.Var(locs[:10, :]),
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=lsl.param(1.0),
            length_scale=lsl.param(1.0),
        )

        model = TransformationModel(y[:, :10], knots=knots.knots, coef=coef)

        assert not jnp.any(jnp.isinf(model.response.value))
        assert model.response.value.shape == (20, 10)

    def test_with_intercept_and_slope(self):
        y = jrd.normal(key, shape=(20, 50))
        locs = jrd.uniform(key, shape=(y.shape[1], 2))

        knots = OnionKnots(-3.0, 3.0, nparam=12)
        coef = OnionCoefPredictivePointProcessGP.new_from_locs(
            knots,
            inducing_locs=lsl.Var(locs[:5, :]),
            sample_locs=lsl.Var(locs),
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=lsl.param(1.0, name="a1"),
            length_scale=lsl.param(1.0, name="l1"),
        )

        intercept = ParamPredictivePointProcessGP(
            inducing_locs=lsl.Var(locs[:5, :]),
            sample_locs=lsl.Var(locs),
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=lsl.param(1.0, name="a2"),
            length_scale=lsl.param(1.0, name="l2"),
            name="intercept",
        )

        slope = ParamPredictivePointProcessGP(
            inducing_locs=lsl.Var(locs[:5, :]),
            sample_locs=lsl.Var(locs),
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=lsl.param(1.0, name="a3"),
            length_scale=lsl.param(1.0, name="l3"),
            name="slope",
        )

        model = TransformationModel(
            y, knots=knots.knots, coef=coef, intercept=intercept, slope=slope
        )
        assert not jnp.any(jnp.isinf(model.response.value))
        assert model.response.value.shape == y.shape

        for name in ["a1", "a2", "a3", "l1", "l2", "l3"]:
            assert name in model.hyperparam_names()

        for name in coef.parameter_names + ["intercept_latent", "slope_latent"]:
            assert name in model.param_names()

    def test_with_simple_transformation(self):
        y = jrd.normal(key, shape=(20, 50))

        knots = OnionKnots(-3.0, 3.0, nparam=12)
        coef = ModelOnionCoef(knots, name="coef")
        model = TransformationModel(y, knots=knots.knots, coef=coef)

        assert not jnp.any(jnp.isinf(model.response.value))
        assert model.response.value.shape == y.shape

        assert model.param_names()[0] == coef.parameter_names[0]

    def test_copy_for(self) -> None:
        y = jrd.normal(key, shape=(20, 50))
        locs = jrd.uniform(key, shape=(y.shape[1], 2))

        knots = OnionKnots(-3.0, 3.0, nparam=12)
        coef = OnionCoefPredictivePointProcessGP.new_from_locs(
            knots,
            inducing_locs=lsl.Var(locs[:5, :]),
            sample_locs=lsl.Var(locs[:10, :]),
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=lsl.param(1.0),
            length_scale=lsl.param(1.0),
        )

        coef.latent_coef.latent_var.value = jrd.normal(
            key, shape=coef.latent_coef.latent_var.value.shape
        )

        model = TransformationModel(y[:, :10], knots=knots.knots, coef=coef)
        z, logdet = model.transformation_and_logdet(y[:, :10])

        model_new = model.copy_for(y, sample_locs=lsl.Var(locs))
        z_new, logdet_new = model_new.transformation_and_logdet(y)

        assert not jnp.allclose(z, y[:, :10], atol=1e-3)

        assert jnp.allclose(model.coef.value, model_new.coef.value[:, :10])

        assert jnp.allclose(z, z_new[:, :10])
        assert jnp.allclose(logdet, logdet_new[:, :10])

        assert z_new.shape == y.shape
        assert logdet_new.shape == y.shape

    def test_fit_batched(self) -> None:
        y = jrd.normal(key, shape=(90, 100))
        locs = jrd.uniform(key, shape=(y.shape[1], 2))

        knots = OnionKnots(-3.0, 3.0, nparam=12)
        coef = OnionCoefPredictivePointProcessGP.new_from_locs(
            knots,
            inducing_locs=lsl.Var(locs[:10, :], name="inducing_locs"),
            sample_locs=lsl.Var(locs, name="locs"),
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=lsl.param(1.0, name="amplitude"),
            length_scale=lsl.param(1.0, name="length_scale"),
            name="coef",
        )

        model = TransformationModel(y[:-10, :], knots=knots.knots, coef=coef)
        model_validation = model.copy_for(y[-10:, :])

        with jax.disable_jit(disable=False):
            model.fit_loc_batched(model_validation=model_validation, loc_batch_size=10)

        assert True


class TestChainedModel:
    @pytest.mark.parametrize(
        "loc,scale", [(0.0, 1.0), (0.0, 2.0), (1.0, 1.0), (1.0, 2.0)]
    )
    def test_init(self, loc, scale):
        y = jrd.normal(key, shape=(20, 50))
        locs = jrd.uniform(key, shape=(y.shape[1], 2))

        knots = OnionKnots(-3.0, 3.0, nparam=12)
        coef = OnionCoefPredictivePointProcessGP.new_from_locs(
            knots,
            inducing_locs=lsl.Var(locs[:5, :]),
            sample_locs=lsl.Var(locs),
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=lsl.param(1.0),
            length_scale=lsl.param(1.0),
        )

        loc_param = MockParam(jnp.full(shape=(y.shape[1],), fill_value=loc))
        scale_param = MockParam(jnp.full(shape=(y.shape[1],), fill_value=scale))

        ChainedModel(
            y=y,
            tfp_dist_cls=tfd.Normal,
            loc=loc_param,
            scale=scale_param,
            knots=knots.knots,
            coef=coef,
        )

        assert True

    @pytest.mark.parametrize(
        "loc,scale", [(0.0, 1.0), (0.0, 2.0), (1.0, 1.0), (1.0, 2.0)]
    )
    def test_init_scalar(self, loc, scale):
        y = jrd.normal(key, shape=(20, 50))
        locs = jrd.uniform(key, shape=(y.shape[1], 2))

        knots = OnionKnots(-3.0, 3.0, nparam=12)
        coef = OnionCoefPredictivePointProcessGP.new_from_locs(
            knots,
            inducing_locs=lsl.Var(locs[:5, :]),
            sample_locs=lsl.Var(locs),
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=lsl.param(1.0),
            length_scale=lsl.param(1.0),
        )

        loc_param = MockParam(loc)
        scale_param = MockParam(scale)

        ChainedModel(
            y=y,
            tfp_dist_cls=tfd.Normal,
            loc=loc_param,
            scale=scale_param,
            knots=knots.knots,
            coef=coef,
        )

        assert True

    @pytest.mark.parametrize(
        "loc,scale", [(0.0, 1.0), (0.0, 2.0), (1.0, 1.0), (1.0, 2.0)]
    )
    def test_transformation(self, loc, scale):
        y = jrd.normal(key, shape=(20, 50))
        locs = jrd.uniform(key, shape=(y.shape[1], 2))

        knots = OnionKnots(-3.0, 3.0, nparam=12)
        coef = OnionCoefPredictivePointProcessGP.new_from_locs(
            knots,
            inducing_locs=lsl.Var(locs[:5, :]),
            sample_locs=lsl.Var(locs),
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=lsl.param(1.0),
            length_scale=lsl.param(1.0),
        )

        loc_param = MockParam(jnp.full(shape=(y.shape[1],), fill_value=loc))
        scale_param = MockParam(jnp.full(shape=(y.shape[1],), fill_value=scale))

        model = ChainedModel(
            y=y,
            tfp_dist_cls=tfd.Normal,
            loc=loc_param,
            scale=scale_param,
            knots=knots.knots,
            coef=coef,
        )

        model.build_graph()

        model2 = Model(
            y=y,
            tfp_dist_cls=tfd.Normal,
            loc=loc_param,
            scale=scale_param,
        )

        z, logdet = model.transformation_and_logdet(y)
        z2, logdet2 = model2.transformation_and_logdet(y)

        assert jnp.allclose(z, z2, atol=1e-5)
        assert jnp.allclose(logdet, logdet2, atol=1e-5)
