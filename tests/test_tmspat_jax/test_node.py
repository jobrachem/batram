from collections.abc import Iterator

import jax.numpy as jnp
import jax.random as jrd
import liesel.model as lsl
import liesel_ptm as ptm
import pytest
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk

import batram.tmspat_jax.node as tm

key = jrd.PRNGKey(42)

locs = jrd.uniform(key, shape=(10,))


class TestKernel:
    def test_shape_2d(self):
        x = jrd.uniform(key, shape=(10, 2))

        amplitude = lsl.param(value=1.0)
        length_scale = lsl.param(value=1.0)

        kernel = tm.Kernel(
            x,
            kernel_class=tfk.ExponentiatedQuadratic,
            amplitude=amplitude,
            length_scale=length_scale,
        )

        kernel.update()

        assert kernel.value.shape == (x.shape[0], x.shape[0])

    def test_1d_error(self):
        x = jrd.uniform(key, shape=(10,))

        amplitude = lsl.param(value=1.0)
        length_scale = lsl.param(value=1.0)

        with pytest.raises(ValueError):
            tm.Kernel(
                x,
                kernel_class=tfk.ExponentiatedQuadratic,
                amplitude=amplitude,
                length_scale=length_scale,
            )

    def test_value_2d(self):
        x = jrd.uniform(key, shape=(10, 2))

        amplitude = lsl.param(value=1.0)
        length_scale = lsl.param(value=1.0)

        kernel = tm.Kernel(
            x,
            kernel_class=tfk.ExponentiatedQuadratic,
            amplitude=amplitude,
            length_scale=length_scale,
        )

        kernel.update()

        tfp_kernel = tfk.ExponentiatedQuadratic(amplitude=1.0, length_scale=1.0)

        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                kernel_ij = kernel.value[i, j]
                tfp_ij = tfp_kernel.apply(x[i, :], x[j, :])
                assert kernel_ij == pytest.approx(tfp_ij)


class TestMultioutputKernel:
    def test_value(self):
        x = jrd.uniform(key, shape=(10, 2))

        amplitude = lsl.param(value=1.0)
        length_scale = lsl.param(value=1.0)

        W = tm.rw_weight_matrix(D=5)

        kernel = tm.MultioutputKernelIMC(
            x,
            W=W,
            kernel_class=tfk.ExponentiatedQuadratic,
            amplitude=amplitude,
            length_scale=length_scale,
        )

        kernel.update()

        assert kernel.value.shape == (40, 40)


def test_brownian_motion_mat():
    B = tm.brownian_motion_mat(3, 3)
    assert B.shape == (3, 3)

    for i in range(3):
        for j in range(3):
            assert B[i, j] == min(i, j) + 1


def test_rw_weight_matrix():
    W = tm.rw_weight_matrix(5)
    assert W.shape == (4, 3)


def test_delta_param():
    nloc = 10
    D = 7
    locs = jrd.uniform(key, shape=(nloc, 2))
    eta = tm.eta_param(locs)
    delta = tm.delta_param(locs, D=D, eta=eta)
    delta.update()

    latent_delta = delta.value_node.kwinputs["latent_delta"].var
    latent_delta.value = jrd.normal(key, shape=((D - 2) * nloc,))
    delta.update()

    assert delta.value.shape == ((D - 1), nloc)
    assert delta.value.mean(axis=0).shape == (nloc,)
    assert jnp.allclose(delta.value.mean(axis=0), 0.0, atol=1e-5)


def test_sfn():
    nloc = 10
    D = 7
    locs = jrd.uniform(key, shape=(nloc, 2))
    eta = tm.eta_param(locs)
    delta = tm.delta_param(locs, D=D, eta=eta)
    delta.update()

    slope_correction_factor = tm.sfn(jnp.exp(delta.value).T, 1.0)
    assert slope_correction_factor.shape == (nloc,)


def test_shape_coef():
    nloc = 10
    D = 7
    locs = jrd.uniform(key, shape=(nloc, 2))
    eta = tm.eta_param(locs)
    delta = tm.delta_param(locs, D=D, eta=eta)
    delta.update()

    shape_coef = tm.shape_coef(delta, 1.0)
    shape_coef.update()

    assert shape_coef.value.shape == delta.value.shape
    assert jnp.all(jnp.diff(shape_coef.value, axis=0) > 0)


def test_validate_shape_coef():
    nloc = 10
    D = 7
    locs = jrd.uniform(key, shape=(nloc, 2))
    eta = tm.eta_param(locs)
    delta = tm.delta_param(locs, D=D, eta=eta)

    latent_delta = delta.value_node.kwinputs["latent_delta"].var
    latent_delta.value = jrd.normal(key, shape=((D - 2) * nloc,))

    delta.update()

    shape_coef = tm.shape_coef(delta, 0.75)
    shape_coef.update()

    for i in range(nloc):
        coef_ptm = ptm.normalization_coef(delta.value[:, i], dknots=0.75)
        assert jnp.allclose(shape_coef.value[:, i], coef_ptm[1:])

    shape_coef = tm.shape_coef(delta, 2.75)
    shape_coef.update()

    for i in range(nloc):
        coef_ptm = ptm.normalization_coef(delta.value[:, i], dknots=2.75)
        assert jnp.allclose(shape_coef.value[:, i], coef_ptm[1:])


def test_alpha_param():
    nloc = 10
    locs = jrd.uniform(key, shape=(nloc, 2))

    alpha_plus_a = tm.alpha_param(locs, knots=jnp.arange(10))
    alpha_plus_a.update()
    assert alpha_plus_a.value.shape == (nloc,)


def test_beta_param():
    nloc = 10
    locs = jrd.uniform(key, shape=(nloc, 2))

    exp_beta = tm.beta_param(locs)
    exp_beta.update()
    assert exp_beta.value.shape == (nloc,)


@pytest.fixture
def model() -> Iterator[tm.Model]:
    nloc = 10
    nobs = 5
    D = 6

    locs = jrd.uniform(key, shape=(nloc, 2))
    y = 2 * jrd.normal(key, (nloc, nobs))
    knots = jnp.linspace(-5, 5, D + 4)

    model = tm.Model(y, knots=knots, locs=locs)
    yield model


class TestModel:
    def test_normalization(self):
        nloc = 10
        nobs = 5
        D = 6

        locs = jrd.uniform(key, shape=(nloc, 2))
        y = 2 * jrd.normal(key, (nloc, nobs))

        knots = jnp.linspace(-5, 5, D + 4)

        model = tm.Model(y, knots=knots, locs=locs)

        assert model.normalization_and_deriv.value[0].shape == y.shape

        fyd = model.normalization_and_deriv.value[1]
        assert jnp.all(fyd > 0.0)

        assert jnp.allclose(model.normalization.value, y, atol=1e-5)

    def test_response_dist(self):
        nloc = 10
        nobs = 5
        D = 6

        locs = jrd.uniform(key, shape=(nloc, 2))
        y = 2 * jrd.normal(key, (nloc, nobs))

        knots = jnp.linspace(-5, 5, D + 4)

        model = tm.Model(y, knots=knots, locs=locs)
        assert model.response.log_prob.shape == (nloc, nobs)

        assert jnp.allclose(
            model.refdist.log_prob(y), model.response.log_prob, atol=1e-5
        )

    def test_eta_param_name(self, model):
        assert model.eta_param_name == "eta"

    def test_eta_hyperparam_names(self, model):
        assert model.eta_hyperparam_names == ["amplitude_eta", "length_scale_eta"]

    def test_delta_param_name(self, model):
        assert model.delta_param_name == "latent_delta"

    def test_delta_hyperparam_names(self, model):
        assert model.delta_hyperparam_names == ["amplitude_delta", "length_scale_delta"]

    def test_alpha_param_name(self, model):
        assert model.alpha_param_name == "alpha"

    def test_alpha_hyperparam_names(self, model):
        assert model.alpha_hyperparam_names == ["amplitude_alpha", "length_scale_alpha"]

    def test_beta_param_name(self, model):
        assert model.beta_param_name == "beta"

    def test_beta_hyperparam_names(self, model):
        assert model.beta_hyperparam_names == ["amplitude_beta", "length_scale_beta"]


def test_predict_normalization():
    nloc = 10
    nobs = 50
    D = 6

    locs = jrd.uniform(key, shape=(nloc, 2))
    y = 2 * jrd.normal(key, (nloc, nobs))

    knots = jnp.linspace(-5, 5, D + 4)

    model = tm.Model(y[:, :10], knots=knots, locs=locs)
    graph = model.build_graph()

    z = tm.predict_normalization(graph, y, graph.state)

    assert z.shape == (nloc, nobs)
    assert jnp.allclose(z, y, atol=1e-5)
