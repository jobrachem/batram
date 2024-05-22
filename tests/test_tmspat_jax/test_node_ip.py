from collections.abc import Iterator

import jax.numpy as jnp
import jax.random as jrd
import liesel.model as lsl
import liesel.goose.optim as optim
import liesel_ptm as ptm
import pytest
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
import jax

import batram.tmspat_jax.node_ip as tm

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
    nloc = 15
    K = 5
    D = 7
    locs = jrd.uniform(key, shape=(nloc, 2))
    eta = tm.EtaParam(locs, K=K)
    delta = tm.DeltaParam(locs, D=D, K=K, eta=eta)
    delta.update()

    latent_delta = delta.value_node.kwinputs["latent_delta"].var
    latent_delta.value = jrd.normal(key, shape=((D - 2) * K,))
    delta.update()

    assert delta.value.shape == ((D - 1), nloc)
    assert delta.value.mean(axis=0).shape == (nloc,)
    assert jnp.allclose(delta.value.mean(axis=0), 0.0, atol=1e-4)


def test_sfn():
    nloc = 15
    K = 5
    D = 7
    locs = jrd.uniform(key, shape=(nloc, 2))
    eta = tm.EtaParam(locs, K=K)
    delta = tm.DeltaParam(locs, D=D, K=K, eta=eta)
    delta.update()

    slope_correction_factor = tm.sfn(jnp.exp(delta.value).T, 1.0)
    assert slope_correction_factor.shape == (nloc,)


def test_shape_coef():
    nloc = 15
    D = 7
    K = 5
    locs = jrd.uniform(key, shape=(nloc, 2))
    eta = tm.EtaParam(locs, K=K)
    delta = tm.DeltaParam(locs, D=D, K=5, eta=eta)
    delta.update()

    shape_coef = tm.ShapeCoef(delta, 1.0)
    shape_coef.update()

    assert shape_coef.value.shape == delta.value.shape
    assert jnp.all(jnp.diff(shape_coef.value, axis=0) > 0)


def test_validate_shape_coef():
    nloc = 15
    K = 5
    D = 7
    locs = jrd.uniform(key, shape=(nloc, 2))
    eta = tm.EtaParam(locs, K=K)
    delta = tm.DeltaParam(locs, D=D, K=K, eta=eta)

    latent_delta = delta.value_node.kwinputs["latent_delta"].var
    latent_delta.value = jrd.normal(key, shape=((D - 2) * K,))

    delta.update()

    shape_coef = tm.ShapeCoef(delta, 0.75)
    shape_coef.update()

    for i in range(nloc):
        coef_ptm = ptm.normalization_coef(delta.value[:, i], dknots=0.75)
        assert jnp.allclose(shape_coef.value[:, i], coef_ptm[1:])

    shape_coef = tm.ShapeCoef(delta, 2.75)
    shape_coef.update()

    for i in range(nloc):
        coef_ptm = ptm.normalization_coef(delta.value[:, i], dknots=2.75)
        assert jnp.allclose(shape_coef.value[:, i], coef_ptm[1:])


def test_alpha_param():
    nloc = 15
    K = 5
    locs = jrd.uniform(key, shape=(nloc, 2))

    alpha_plus_a = tm.AlphaParam(locs, knots=jnp.arange(10), K=5)
    alpha_plus_a.update()
    assert alpha_plus_a.value.shape == (nloc,)


def test_beta_param():
    nloc = 15
    K = 5
    locs = jrd.uniform(key, shape=(nloc, 2))

    exp_beta = tm.ExpBetaParam(locs, K=K)
    exp_beta.update()
    assert exp_beta.value.shape == (nloc,)


@pytest.fixture
def model() -> Iterator[tm.Model]:
    nloc = 15
    K = 5
    nobs = 5
    D = 6

    locs = jrd.uniform(key, shape=(nloc, 2))
    y = 2 * jrd.normal(key, (nobs, nloc))
    knots = jnp.linspace(-5, 5, D + 4)

    model = tm.Model(y, knots=knots, locs=locs, K=K)
    yield model


class TestModel:
    def test_normalization(self):
        nloc = 15
        K = 5
        nobs = 5
        D = 6

        locs = jrd.uniform(key, shape=(nloc, 2))
        y = 2 * jrd.normal(key, (nobs, nloc))

        knots = jnp.linspace(-5, 5, D + 4)

        model = tm.Model(y, knots=knots, locs=locs, K=K)

        assert model.normalization_and_deriv.value[0].T.shape == y.shape

        fyd = model.normalization_and_deriv.value[1]
        assert jnp.all(fyd > 0.0)

        assert jnp.allclose(model.normalization.value, y, atol=1e-5)

    def test_response_dist(self):
        nloc = 15
        K = 5
        nobs = 5
        D = 6

        locs = jrd.uniform(key, shape=(nloc, 2))
        y = 2 * jrd.normal(key, (nobs, nloc))

        knots = jnp.linspace(-5, 5, D + 4)

        model = tm.Model(y, knots=knots, locs=locs, K=K)
        assert model.response.log_prob.shape == (nobs, nloc)

        assert jnp.allclose(
            model.refdist.log_prob(y), model.response.log_prob, atol=1e-5
        )

    def test_eta_param_name(self, model):
        assert model.eta.parameter_names == ["latent_eta"]

    def test_eta_hyperparam_names(self, model):
        assert model.eta.hyperparameter_names == [
            "amplitude_eta_transformed",
            "length_scale_eta_transformed",
        ]

    def test_delta_param_name(self, model):
        assert model.delta.parameter_names == ["latent_delta"]

    def test_delta_hyperparam_names(self, model):
        assert model.delta.hyperparameter_names == [
            "amplitude_delta_transformed",
            "length_scale_delta_transformed",
        ]

    def test_alpha_param_name(self, model):
        assert model.alpha.parameter_names == ["latent_alpha"]

    def test_alpha_hyperparam_names(self, model):
        assert model.alpha.hyperparameter_names == [
            "amplitude_alpha_transformed",
            "length_scale_alpha_transformed",
        ]

    def test_beta_param_name(self, model):
        assert model.exp_beta.parameter_names == ["latent_beta"]

    def test_beta_hyperparam_names(self, model):
        assert model.exp_beta.hyperparameter_names == [
            "amplitude_beta_transformed",
            "length_scale_beta_transformed",
        ]


def test_predict_normalization():
    nloc = 15
    nobs = 50
    D = 6
    K = 5

    locs = jrd.uniform(key, shape=(nloc, 2))
    y = 2 * jrd.normal(key, (nobs, nloc))

    knots = jnp.linspace(-5, 5, D + 4)

    with jax.disable_jit():
        model = tm.Model(y[:10, :], knots=knots, locs=locs, K=K)
    graph = model.build_graph()

    z, z_deriv = tm.predict_normalization_and_deriv(graph, y, graph.state)

    assert z.shape == (nobs, nloc)
    assert z_deriv.shape == (nobs, nloc)

    assert jnp.allclose(z, y, atol=1e-5)
    assert jnp.allclose(z_deriv, 1.0, atol=1e-5)


def test_optim():
    nloc = 1
    nobs = 500
    D = 15

    locs = jrd.uniform(key, shape=(nloc, 2))
    y = 2 * jrd.exponential(key, (nobs, nloc))

    knots_lo = jnp.quantile(y, 0.01)
    knots_hi = jnp.quantile(y, 0.99)
    knots = ptm.kn(jnp.array([knots_lo, knots_hi]), order=3, n_params=D)

    model = tm.Model(y[:300, :], knots=knots, locs=locs, K=1)
    graph = model.build_graph()

    params = (
        model.alpha.parameter_names
        + model.exp_beta.parameter_names
        + model.delta.parameter_names
        + model.eta.parameter_names
    )
    hyperparams = (
        model.alpha.hyperparameter_names
        + model.exp_beta.hyperparameter_names
        + model.delta.hyperparameter_names
        + model.eta.hyperparameter_names
    )

    optim_params = params + hyperparams
    stopper = optim.Stopper(max_iter=1000, patience=10)
    result = optim.optim_flat(graph, optim_params, stopper=stopper)
    assert result.iteration == 1000

    z_init, _ = tm.predict_normalization_and_deriv(graph, y, graph.state)
    z_fit, _ = tm.predict_normalization_and_deriv(graph, y, result.model_state)

    assert not jnp.allclose(z_init, z_fit, atol=1e-2)
