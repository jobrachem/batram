import jax.numpy as jnp
import jax.random as jrd
import veccs.orderings

import batram.tmspat_jax.node as tm
import batram.tmspat_jax.node_ip as tmip

key = jrd.PRNGKey(42)


class TestSpeed:
    def test_delta(self):
        Nloc = 100

        locs = jrd.uniform(key, shape=(Nloc, 2))
        order = veccs.orderings.maxmin_cpp(locs)
        locs = locs[order, :]

        K = Nloc // 10
        nobs = 5
        D = 15

        y = 2 * jrd.normal(key, (locs.shape[0], nobs))
        knots = jnp.linspace(-5, 5, D + 4)

        model_ip = tmip.Model(y, knots=knots, locs=locs, K=K)
        model = tm.Model(y, knots=knots, locs=locs)

        def update_delta_ip():
            model_ip.delta.latent.value = jrd.normal(key, ((D - 2) * K,))
            model_ip.delta.update()

        def update_delta():
            latent_delta = model.delta.value_node.kwinputs["latent_delta"].var
            latent_delta.value = jrd.normal(key, ((D - 2) * locs.shape[0],))
            model.delta.update()

        # time_ip = timeit(update_delta_ip, number=100)
        # time_exact = timeit(update_delta, number=100)

    def test_model(self):
        Nloc = 100

        locs = jrd.uniform(key, shape=(Nloc, 2))
        order = veccs.orderings.maxmin_cpp(locs)
        locs = locs[order, :]

        K = Nloc // 10
        nobs = 50
        D = 15

        y = 2 * jrd.normal(key, (locs.shape[0], nobs))
        knots = jnp.linspace(-5, 5, D + 4)

        model_ip = tmip.Model(y, knots=knots, locs=locs, K=K)
        model = tm.Model(y, knots=knots, locs=locs)

        def update_delta_ip():
            latent_delta = model_ip.delta.latent
            graph = model_ip.build_graph()
            latent_delta.value = jrd.normal(key, ((D - 2) * K,))
            graph.update()

        def update_delta():
            latent_delta = model.delta.value_node.kwinputs["latent_delta"].var
            graph = model.build_graph()
            latent_delta.value = jrd.normal(key, ((D - 2) * locs.shape[0],))
            graph.update()

        # time_ip = timeit(update_delta_ip, number=10)
        # time_exact = timeit(update_delta, number=10)

        assert True
