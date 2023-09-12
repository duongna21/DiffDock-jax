from typing import Optional

import jax
import jax.numpy as jnp
from functools import partial
import jax.lax
from jax.experimental import checkify

@partial(jax.jit, static_argnums=5)
def radius_helper(
    x: jnp.ndarray,
    y: jnp.ndarray,
    ptr_x: jnp.ndarray = None,
    ptr_y: jnp.ndarray = None,
    r=1.0,
    max_num_neighbors=32):

    n, m = x.shape[0], y.shape[0]

    if ptr_x is None:
        ptr_x = jnp.arange(0, n + 1, n)
    if ptr_y is None:
        ptr_y = jnp.arange(0, m + 1, m)

    row = -jnp.ones(m * max_num_neighbors)
    col = -jnp.ones(m * max_num_neighbors)

    r_squared = r ** 2

    def outer_loop(n_y, carry):
        count = 0
        row, col = carry
        example_idx = jnp.sum(ptr_y <= n_y) - 1
        start, end = ptr_x[example_idx], ptr_x[example_idx + 1]

        def inner_loop(n_x, carry):
            count, row, col = carry
            dist = jnp.sum((x[n_x, :] - y[n_y, :]) ** 2)
            condition = (dist < r_squared) & (count < max_num_neighbors)
            row = jax.lax.cond(condition, lambda _: row.at[n_y * max_num_neighbors + count].set(n_y), lambda _: row, None)
            col = jax.lax.cond(condition, lambda _: col.at[n_y * max_num_neighbors + count].set(n_x), lambda _: col, None)
            count = jax.lax.cond(condition, lambda _: count + 1, lambda _: count, None)
            return count, row, col
        count, row, col = jax.lax.fori_loop(start, end, inner_loop, (count, row, col))
        return row, col

    row, col = jax.lax.fori_loop(0, m, outer_loop, (row, col))

    mask = row != -1
    row = jnp.where(mask, row, 0)
    col = jnp.where(mask, col, 0)
    return jnp.stack([row, col], axis=0)


# @partial(jax.jit, static_argnames=['max_num_neighbors', 'batch_size'])
def radius(
    x: jnp.ndarray,
    y: jnp.ndarray,
    r: float,
    batch_x: Optional[jnp.ndarray] = None,
    batch_y: Optional[jnp.ndarray] = None,
    max_num_neighbors: int = 32,
    batch_size: Optional[int] = None,
) -> jnp.ndarray:

    if x.size == 0 or y.size == 0:
        return jnp.empty((2, 0), dtype=jnp.int32)

    x = x.reshape((-1, 1)) if x.ndim == 1 else x
    y = y.reshape((-1, 1)) if y.ndim == 1 else y

    if batch_size is None:
        batch_size = 1
        if batch_x is not None:
            assert x.shape[0] == batch_x.size
            batch_size = batch_x.max().astype(int) + 1
        if batch_y is not None:
            assert y.shape[0] == batch_y.size
            batch_size = jnp.maximum(batch_size, batch_y.max().astype(int) + 1)
    assert batch_size > 0
    ptr_x: Optional[jnp.ndarray] = None
    ptr_y: Optional[jnp.ndarray] = None

    if batch_size > 1:
        assert batch_x is not None
        assert batch_y is not None
        arange = jnp.arange(batch_size + 1)
        ptr_x = jnp.digitize(arange, batch_x, right=True)
        ptr_y = jnp.digitize(arange, batch_y, right=True)

    return radius_helper(x, y, ptr_x, ptr_y, r, max_num_neighbors)


def radius_graph(
    x: jnp.ndarray,
    r: float,
    batch: Optional[jnp.ndarray] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = 'source_to_target',
    batch_size: Optional[int] = None,
) -> jnp.ndarray:

    assert flow in ['source_to_target', 'target_to_source']

    # Assume there's a JAX equivalent for the `radius` function.
    edge_index = radius(x, x, r, batch, batch,
                        max_num_neighbors if loop else max_num_neighbors + 1, batch_size)
    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]

    return jnp.stack([row, col], axis=0)