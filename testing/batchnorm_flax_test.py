import jax.numpy as jnp
import pytest
import jax
from batchnorm_flax import BatchNorm
import e3nn_jax as e3nn
from e3nn_jax.utils import assert_equivariant

@pytest.mark.parametrize("irreps", [e3nn.Irreps("3x0e + 3x0o + 4x1e"), e3nn.Irreps("3x0o + 3x0e + 4x1e")])
@pytest.mark.parametrize("keys", [jax.random.PRNGKey(1), jax.random.PRNGKey(15)])

def test_equivariant(irreps, keys):
    model = BatchNorm(irreps=irreps)
    x = e3nn.normal(irreps, keys, (16,))
    params = model.init(keys, x)
    
    # Apply model
    x = e3nn.normal(irreps, keys, (16,))
    _ = model.apply(params, x, mutable=['batch_stats'])
    
    x = e3nn.normal(irreps, keys, (16,))
    _ = model.apply(params, x, mutable=['batch_stats'])

    m_train = lambda x: model.apply(params, x, mutable=['batch_stats'])[0]
    assert_equivariant(m_train, keys, e3nn.normal(irreps, keys, (16,)))
    
    m_eval = lambda x: model.apply(params, x, is_training=False, mutable=['batch_stats'])[0]
    assert_equivariant(m_eval, keys, e3nn.normal(irreps, keys, (16,)))


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("reduce", ["mean", "max"])
@pytest.mark.parametrize("normalization", ["norm", "component"])
@pytest.mark.parametrize("instance", [True, False])
@pytest.mark.parametrize("keys", [jax.random.PRNGKey(1), jax.random.PRNGKey(15)])
def test_modes(keys, affine, reduce, normalization, instance):

    irreps = e3nn.Irreps("10x0e + 5x1e")
    
    model = BatchNorm(
            irreps=irreps,
            affine=affine,
            reduce=reduce,
            normalization=normalization,
            instance=instance,
        )
    params = model.init(keys, e3nn.normal(irreps, keys, (20,20)))
    
    m_train = lambda x: model.apply(params, x, mutable=['batch_stats'])[0]
    m_eval = lambda x: model.apply(params, x, mutable=['batch_stats'], is_training=False)[0]

    m_train(e3nn.normal(irreps, keys, (20, 20)))

    m_eval(e3nn.normal(irreps, keys, (20, 20)))


@pytest.mark.parametrize("instance", [True, False])
@pytest.mark.parametrize("keys", [jax.random.PRNGKey(8), jax.random.PRNGKey(6)])
def test_normalization(keys, instance):
    float_tolerance = 1e-3
    sqrt_float_tolerance = jnp.sqrt(float_tolerance)

    batch, n = 20, 20
    irreps = e3nn.Irreps("3x0e + 4x1e")

    model = BatchNorm(irreps=irreps, normalization="norm", instance=instance)

    params = model.init(keys, e3nn.normal(irreps, keys, (16,)))

    x = e3nn.normal(irreps, keys, (batch, n)) * 5
    x,_ = model.apply(params, x, mutable=['batch_stats'])
    a = x.chunks[0]  # [batch, space, mul, 1]
    assert jnp.max(jnp.abs(a.mean([0, 1]))) < float_tolerance
    assert jnp.max(jnp.abs(jnp.square(a).mean([0, 1]) - 1)) < sqrt_float_tolerance

    a = x.chunks[1]  # [batch, space, mul, repr]
    assert (
        jnp.max(jnp.abs(jnp.square(a).sum(3).mean([0, 1]) - 1)) < sqrt_float_tolerance
    )

    model = BatchNorm(irreps=irreps, normalization="component", instance=instance)

    params = model.init(keys, e3nn.normal(irreps, keys, (16,)))

    params = model.init(keys, e3nn.normal(irreps, keys, (16,)))

    x = e3nn.normal(irreps, keys, (batch, n)) * 5
    x, state = model.apply(params, x, mutable=['batch_stats'])

    a = x.chunks[0]  # [batch, space, mul, 1]
    assert jnp.max(jnp.abs(a.mean([0, 1]))) < float_tolerance
    assert jnp.max(jnp.abs(jnp.square(a).mean([0, 1]) - 1)) < sqrt_float_tolerance

    a = x.chunks[1]  # [batch, space, mul, repr]
    assert (
        jnp.max(jnp.abs(jnp.square(a).mean(3).mean([0, 1]) - 1)) < sqrt_float_tolerance
    )