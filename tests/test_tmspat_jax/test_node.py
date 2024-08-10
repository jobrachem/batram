import jax.numpy as jnp
import jax.random as jrd
import liesel.model as lsl
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
from liesel_ptm.bsplines import OnionKnots

import batram.tmspat_jax.node as node

key = jrd.PRNGKey(42)


def exponentiated_quadratic_kernel(dist, amplitude, length_scale):
    cov = jnp.zeros((dist.shape[0], dist.shape[1]))

    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            cov = cov.at[i, j].set(
                amplitude**2 * jnp.exp(-dist[i, j] ** 2 / (2 * length_scale**2))
            )

    return cov


def matrix_of_distances(x1, x2):
    dists = jnp.zeros((x1.shape[0], x2.shape[0]))

    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            dists = dists.at[i, j].set(jnp.linalg.norm(x1[i, :] - x2[j, :]))

    return dists


class TestKernel:
    def test_2dloc(self):
        amplitude = lsl.param(1.0)
        length_scale = lsl.param(1.0)

        locs = lsl.Var(jrd.uniform(key, shape=(10, 2)))

        kernel = node.Kernel(
            locs,
            locs,
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=amplitude,
            length_scale=length_scale,
        )

        assert kernel.value.shape == (10, 10)

        dist = matrix_of_distances(locs.value, locs.value)
        cov = exponentiated_quadratic_kernel(
            dist, amplitude=amplitude.value, length_scale=length_scale.value
        )

        assert jnp.allclose(cov, kernel.value)

    def test_3dloc(self):
        amplitude = lsl.param(1.0)
        length_scale = lsl.param(1.0)

        locs = lsl.Var(jrd.uniform(key, shape=(10, 3)))

        kernel = node.Kernel(
            locs,
            locs,
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=amplitude,
            length_scale=length_scale,
        )

        assert kernel.value.shape == (10, 10)

        dist = matrix_of_distances(locs.value, locs.value)
        cov = exponentiated_quadratic_kernel(
            dist, amplitude=amplitude.value, length_scale=length_scale.value
        )

        assert jnp.allclose(cov, kernel.value)

    def test_2dloc_subset(self):
        amplitude = lsl.param(1.0)
        length_scale = lsl.param(1.0)

        locs = lsl.Var(jrd.uniform(key, shape=(10, 2)))

        kernel = node.Kernel(
            locs,
            locs,
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=amplitude,
            length_scale=length_scale,
        )

        assert kernel.value.shape == (10, 10)

        locs_subset = lsl.Var(locs.value[:5, :])

        kernel2 = node.Kernel(
            locs,
            locs_subset,
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=amplitude,
            length_scale=length_scale,
        )

        assert kernel2.value.shape == (10, 5)


class TestRandomWalkParamPredictivePointGP:
    def test_init(self):
        amplitude = lsl.param(1.0)
        length_scale = lsl.param(1.0)

        locs = jrd.uniform(key, shape=(30, 2))

        param = node.RandomWalkParamPredictivePointProcessGP(
            inducing_locs=lsl.Var(locs[:5, :]),
            sample_locs=lsl.Var(locs),
            D=10,
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=amplitude,
            length_scale=length_scale,
        )

        assert not jnp.any(jnp.isinf(param.value))

        assert param.value.shape == (9, 30)


class TestOnionCoefPredictivePointProcessGP:
    def test_init(self):
        amplitude = lsl.param(1.0)
        length_scale = lsl.param(1.0)

        locs = jrd.uniform(key, shape=(30, 2))

        knots = OnionKnots(a=-3.0, b=3.0, nparam=10)

        param = node.OnionCoefPredictivePointProcessGP.new_from_locs(
            knots=knots,
            inducing_locs=lsl.Var(locs[:5, :]),
            sample_locs=lsl.Var(locs),
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=amplitude,
            length_scale=length_scale,
        )

        assert not jnp.any(jnp.isinf(param.value))
        assert param.value.shape == (knots.nparam + 6 + 1, 30)

    def test_spawn_intercept(self):
        amplitude = lsl.param(1.0)
        length_scale = lsl.param(1.0)

        locs = jrd.uniform(key, shape=(30, 2))

        knots = OnionKnots(a=-3.0, b=3.0, nparam=10)

        param = node.OnionCoefPredictivePointProcessGP.new_from_locs(
            knots=knots,
            inducing_locs=lsl.Var(locs[:5, :]),
            sample_locs=lsl.Var(locs),
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=amplitude,
            length_scale=length_scale,
        )

        intercept = param.spawn_intercept()

        assert amplitude in list(intercept.kernel_params.values())
        assert length_scale in list(intercept.kernel_params.values())

    def test_spawn_slope(self):
        amplitude = lsl.param(1.0)
        length_scale = lsl.param(1.0)

        locs = jrd.uniform(key, shape=(30, 2))

        knots = OnionKnots(a=-3.0, b=3.0, nparam=10)

        param = node.OnionCoefPredictivePointProcessGP.new_from_locs(
            knots=knots,
            inducing_locs=lsl.Var(locs[:5, :]),
            sample_locs=lsl.Var(locs),
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=amplitude,
            length_scale=length_scale,
        )

        slope = param.spawn_slope()

        assert amplitude in list(slope.kernel_params.values())
        assert length_scale in list(slope.kernel_params.values())
        assert jnp.all(slope.value > 0.0)

    def test_copy_for(self):
        amplitude = lsl.param(1.0)
        length_scale = lsl.param(1.0)

        locs = jrd.uniform(key, shape=(30, 2))

        knots = OnionKnots(a=-3.0, b=3.0, nparam=10)

        param = node.OnionCoefPredictivePointProcessGP.new_from_locs(
            knots=knots,
            inducing_locs=lsl.Var(locs[:5, :]),
            sample_locs=lsl.Var(locs[:10, :]),
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=amplitude,
            length_scale=length_scale,
        )

        param.latent_coef.latent_var.value = jrd.uniform(
            key, shape=param.latent_coef.latent_var.value.shape
        )

        param_new = param.copy_for(lsl.Var(locs))

        assert jnp.allclose(param_new.value[:, :10], param.value)
        assert param_new.value.shape == (knots.nparam + 7, locs.shape[0])

        assert param_new.latent_coef.kernel_params["amplitude"] is not amplitude
        assert param_new.latent_coef.kernel_params["length_scale"] is not length_scale


class TestParamPredictivePointProcessGP:
    def test_copy_for(self):
        amplitude = lsl.param(1.0)
        length_scale = lsl.param(1.0)

        locs = jrd.uniform(key, shape=(30, 2))

        param = node.ParamPredictivePointProcessGP(
            inducing_locs=lsl.Var(locs[:5, :]),
            sample_locs=lsl.Var(locs[:10, :]),
            kernel_cls=tfk.ExponentiatedQuadratic,
            amplitude=amplitude,
            length_scale=length_scale,
        )

        param.latent_var.value = jrd.uniform(key, shape=param.latent_var.value.shape)

        param_new = param.copy_for(lsl.Var(locs))

        assert jnp.allclose(param_new.value[:10], param.value)
        assert param_new.value.shape == (locs.shape[0],)

        assert param_new.kernel_params["amplitude"] is not amplitude
        assert param_new.kernel_params["length_scale"] is not length_scale
