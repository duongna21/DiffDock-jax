from functools import partial
import jax
import jax.numpy as jnp
from jax import vmap
from typing import Optional, Tuple


def broadcast(src: jnp.ndarray, other: jnp.ndarray, dim: int) -> jnp.ndarray:
    if dim < 0:
        dim = other.ndim + dim
    if src.ndim == 1:
        for _ in range(0, dim):
            src = jnp.expand_dims(src, 0)
    for _ in range(src.ndim, other.ndim):
        src = jnp.expand_dims(src, -1)
    src = jnp.broadcast_to(src, other.shape)
    return src


def scatter_helper(input, dim, index, src, reduce=None):
    # JAX-port of PyTorch's scatter. See https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html
    # One can simplify this implementation with ndarray.at (ref: https://github.com/google/jax/issues/8487)
    try:
        dnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,))
        
        if reduce is None:
            _scatter = jax.lax.scatter
        elif reduce == "add":
            _scatter = jax.lax.scatter_add
        elif reduce == "multiply":
            _scatter = jax.lax.scatter_mul
            
        _scatter = partial(_scatter, dimension_numbers=dnums)
        vmap_inner = partial(vmap, in_axes=(0, 0, 0), out_axes=0)

        for _ in range(len(input.shape)-1):
            _scatter = vmap_inner(_scatter)
        swap = lambda x: jnp.swapaxes(x, dim, -1)
        input, index, src = list(map(swap, (input, index, src)))
        return swap(_scatter(input, jnp.expand_dims(index, axis=-1), src))
    except:
        return jnp.array(0)


def scatter_sum(src: jnp.ndarray, index: jnp.ndarray, dim: int = -1,
                out: Optional[jnp.ndarray] = None,
                dim_size: Optional[int] = None) -> jnp.ndarray:
    
    index = broadcast(index, src, dim)

    if out is None:
        size = list(src.shape)
        if dim_size is not None:
            size[dim] = dim_size
        elif index.size == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = jnp.zeros(size, dtype=src.dtype)
    
    return scatter_helper(out, dim, index, src, reduce='add')

def scatter_mean(src: jnp.ndarray, index: jnp.ndarray, dim: int = -1,
                 out: Optional[jnp.ndarray] = None,
                 dim_size: Optional[int] = None) -> jnp.ndarray:

    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.shape[dim]

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.ndim
    if index.ndim <= index_dim:
        index_dim = index.ndim - 1

    ones = jnp.ones(index.shape, dtype=src.dtype)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count = jnp.where(count < 1, 1, count)
    count = broadcast(count, out, dim)
    if out.dtype.kind in {'f', 'c'}:  # Checking for floating point or complex type
        return jnp.true_divide(out, count)
    else:
        return jnp.floor(out / count)
    # return out / count

def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
    if reduce == 'sum' or reduce == 'add':
        return scatter_sum(src, index, dim, out, dim_size)
    elif reduce == 'mean':
        return scatter_mean(src, index, dim, out, dim_size)
    else:
        raise ValueError("The specified reduction is not supported.")
    
    
