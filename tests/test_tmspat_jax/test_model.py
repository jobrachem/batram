import liesel.model as lsl
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
from batram.tmspat_jax.model import Model, TransformationModel, ChainedModel
import jax.numpy as jnp
import jax.random as jrd
import jax
import pytest
from batram.tmspat_jax.ppnode import OnionCoefPredictivePointProcessGP, OnionKnots, RandomWalkParamPredictivePointProcessGP, OnionCoef, TransformedVar
import batram.tmspat_jax.model_onion2 as mo2

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
        y = jrd.normal(key, shape=(20,50))

        loc = MockParam(jnp.zeros(50), name="loc")
        scale = MockParam(jnp.ones(50), name="scale")

        Model(y=y, tfp_dist_cls=tfd.Normal, loc=loc, scale=scale)
        assert True
    
    @pytest.mark.parametrize("loc,scale",[(0.0, 1.0), (0.0, 2.0), (1.0, 1.0), (1.0, 2.0)])
    def test_transformation_and_logdet_normal(self, loc, scale):
        y = jrd.normal(key, shape=(20,))

        model = Model(
            y=y, tfp_dist_cls=tfd.Normal, loc=MockParam(loc), scale=MockParam(scale)
        )

        z, logdet = model.transformation_and_logdet(y)

        assert jnp.allclose((y - loc) / scale, z)
        assert jnp.allclose(logdet, -jnp.log(scale), atol=1e-6)

    @pytest.mark.parametrize("loc,scale",[(0.0, 1.0), (0.0, 2.0), (1.0, 1.0), (1.0, 2.0)])
    def test_transformation_and_logdet_normal_locs(self, loc, scale):
        y = jrd.normal(key, shape=(20,50))

        loc_param = MockParam(jnp.full(shape=(50,), fill_value=loc))
        scale_param = MockParam(jnp.full(shape=(50,), fill_value=scale))
        model = Model(
            y=y, tfp_dist_cls=tfd.Normal, loc=loc_param, scale=scale_param
        )

        z, logdet = model.transformation_and_logdet(y)
        
        assert z.shape == y.shape

        assert jnp.allclose((y - loc_param.value) / scale_param.value, z, atol=1e-5)
        assert jnp.allclose(logdet, -jnp.log(scale_param.value), atol=1e-4)
    
    @pytest.mark.parametrize("loc,scale",[(0.0, 1.0), (0.0, 2.0), (1.0, 1.0), (1.0, 2.0)])
    def test_transformation_inverse_normal(self, loc, scale):
        y = jrd.normal(key, shape=(20,))

        model = Model(
            y=y, tfp_dist_cls=tfd.Normal, loc=MockParam(loc), scale=MockParam(scale)
        )

        z = jnp.linspace(-2.0, 2.0, 30)

        ynew = model.transformation_inverse(z)

        assert jnp.allclose(z * scale + loc, ynew)
    
    @pytest.mark.parametrize("loc,scale",[(0.0, 1.0), (0.0, 2.0), (1.0, 1.0), (1.0, 2.0)])
    def test_transformation_inverse_normal_locs(self, loc, scale):
        y = jrd.normal(key, shape=(20,50))

        loc_param = MockParam(jnp.full(shape=(50,), fill_value=loc))
        scale_param = MockParam(jnp.full(shape=(50,), fill_value=scale))
        model = Model(
            y=y, tfp_dist_cls=tfd.Normal, loc=loc_param, scale=scale_param
        )

        z = jnp.linspace(-2.0, 2.0, 50)
        z = jnp.c_[z, z].T
        ynew = model.transformation_inverse(z)
        
        assert ynew.shape == (2, 50)

        assert jnp.allclose(ynew, z * scale_param.value + loc_param.value, atol=1e-5)


class TestTransformationModel:

    def test_against_model_onion2(self):
        y = jrd.normal(key, shape=(20, 50))
        locs = jrd.uniform(key, shape=(y.shape[1], 2))

        knots = OnionKnots(-3.0, 3.0, nparam=12)
        latent_coef = RandomWalkParamPredictivePointProcessGP(
            locs=locs, 
            D=knots.nparam + 1,
            K=5,
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=TransformedVar(jax.nn.softplus(0.5), name="amplitude"),
            length_scale=TransformedVar(jax.nn.softplus(0.5), name="length_scale"),
        )

        coef_spec = OnionCoef(knots)
        coef = OnionCoefPredictivePointProcessGP(latent_coef, coef_spec)

        model = TransformationModel(y, knots=knots.knots, coef=coef)
        model.build_graph()

        model_old = mo2.Model(
            y=y,
            knots=knots,
            coef_spec=coef_spec,
            locs=locs,
            K=5,
            smoothing_prior=mo2.DeltaSmoothing.RANDOM_WALK,
            kernel_class=tfk.ExponentiatedQuadratic,
            eta_fixed=True,
        )
        model_old.build_graph()

        assert jnp.allclose(model.transformation.value, y, atol=1e-2)
        assert jnp.allclose(model_old.normalization.value, y, atol=1e-2)

        assert jnp.allclose(model.response.log_prob , model_old.response.log_prob)

        new_latent = jrd.normal(key=key, shape=latent_coef.latent_var.value.shape)
        latent_coef.latent_var.value = new_latent
        model_old.delta.latent_var.value = new_latent
        
        model.graph.update()
        model_old.graph.update()

        assert jnp.allclose(latent_coef.kernel_du.value, model_old.delta.kernel_du.value, atol=1e-5)
        assert jnp.allclose(latent_coef.kernel_uu.value, model_old.delta.kernel_uu.value, atol=1e-5)
        assert jnp.allclose(latent_coef.W, model_old.delta.W)
        assert jnp.allclose(latent_coef.value, model_old.delta.value, atol=1e-5)
        assert jnp.allclose(model.coef.value, model_old.coef.value.T, atol=1e-5)

        assert jnp.allclose(model.graph.log_prob, model_old.graph.log_prob)
        
        
        
        

    
    def test_init(self):
        y = jrd.normal(key, shape=(20, 50))
        locs = jrd.uniform(key, shape=(y.shape[1], 2))

        knots = OnionKnots(-3.0, 3.0, nparam=12)
        coef = OnionCoefPredictivePointProcessGP.new_from_locs(
            knots, 
            locs,
            K=5,
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=lsl.param(1.0),
            length_scale=lsl.param(1.0),
        )

        model = TransformationModel(y, knots=knots.knots, coef=coef)

        assert not jnp.any(jnp.isinf(model.response.value))
        assert model.response.value.shape == y.shape




class TestChainedModel:

    @pytest.mark.parametrize("loc,scale",[(0.0, 1.0), (0.0, 2.0), (1.0, 1.0), (1.0, 2.0)])
    def test_init(self, loc, scale):
        y = jrd.normal(key, shape=(20, 50))
        locs = jrd.uniform(key, shape=(y.shape[1], 2))
        
        knots = OnionKnots(-3.0, 3.0, nparam=12)
        coef = OnionCoefPredictivePointProcessGP.new_from_locs(
            knots, 
            locs,
            K=5,
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

    @pytest.mark.parametrize("loc,scale",[(0.0, 1.0), (0.0, 2.0), (1.0, 1.0), (1.0, 2.0)])
    def test_init_scalar(self, loc, scale):
        y = jrd.normal(key, shape=(20, 50))
        locs = jrd.uniform(key, shape=(y.shape[1], 2))
        
        knots = OnionKnots(-3.0, 3.0, nparam=12)
        coef = OnionCoefPredictivePointProcessGP.new_from_locs(
            knots, 
            locs,
            K=5,
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

    @pytest.mark.parametrize("loc,scale",[(0.0, 1.0), (0.0, 2.0), (1.0, 1.0), (1.0, 2.0)])
    def test_transformation(self, loc, scale):
        y = jrd.normal(key, shape=(20, 50))
        locs = jrd.uniform(key, shape=(y.shape[1], 2))
        
        knots = OnionKnots(-3.0, 3.0, nparam=12)
        coef = OnionCoefPredictivePointProcessGP.new_from_locs(
            knots, 
            locs,
            K=5,
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