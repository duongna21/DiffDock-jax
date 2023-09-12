import numpy as np
import tqdm
import os
import jax.numpy as jnp
import jax
"""
    Preprocessing for the SO(2)/torus sampling and score computations, truncated infinite series are computed and then
    cached to memory, therefore the precomputation is only run the first time the repository is run on a machine
"""

def p(x, sigma, N=10):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += jnp.exp(-(x + 2 * jnp.pi * i) ** 2 / 2 / sigma ** 2)
    return p_


def grad(x, sigma, N=10):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += (x + 2 * jnp.pi * i) / sigma ** 2 * jnp.exp(-(x + 2 * jnp.pi * i) ** 2 / 2 / sigma ** 2)
    return p_


X_MIN, X_N = 1e-5, 5000  # relative to pi
SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000  # relative to pi

x = 10 ** jnp.linspace(jnp.log10(X_MIN), 0, X_N + 1) * jnp.pi
sigma = 10 ** jnp.linspace(jnp.log10(SIGMA_MIN), jnp.log10(SIGMA_MAX), SIGMA_N + 1) * jnp.pi

if os.path.exists('.p.npy'):
    p_ = jnp.load('.p.npy')
    score_ = jnp.load('.score.npy')
else:
    print("Precomputing and saving to cache torus distribution table")
    p_ = p(x, sigma[:, None], N=100)
    jnp.save('.p.npy', p_)

    score_ = grad(x, sigma[:, None], N=100) / p_
    jnp.save('.score.npy', score_)


def score(x, sigma):
    x = (x + jnp.pi) % (2 * jnp.pi) - jnp.pi
    sign = jnp.sign(x)
    x = jnp.log(jnp.abs(x) / jnp.pi)
    x = (x - jnp.log(X_MIN)) / (0 - jnp.log(X_MIN)) * X_N
    x = jnp.round(jnp.clip(x, 0, X_N)).astype(int)
    sigma = jnp.log(sigma / jnp.pi)
    sigma = (sigma - jnp.log(SIGMA_MIN)) / (jnp.log(SIGMA_MAX) - jnp.log(SIGMA_MIN)) * SIGMA_N
    sigma = jnp.round(jnp.clip(sigma, 0, SIGMA_N)).astype(int)
    return -sign * score_[sigma, x]


def p(x, sigma):
    x = (x + jnp.pi) % (2 * jnp.pi) - jnp.pi
    x = jnp.log(jnp.abs(x) / jnp.pi)
    x = (x - jnp.log(X_MIN)) / (0 - jnp.log(X_MIN)) * X_N
    x = jnp.round(jnp.clip(x, 0, X_N)).astype(int)
    sigma = jnp.log(sigma / jnp.pi)
    sigma = (sigma - jnp.log(SIGMA_MIN)) / (jnp.log(SIGMA_MAX) - jnp.log(SIGMA_MIN)) * SIGMA_N
    sigma = jnp.round(jnp.clip(sigma, 0, SIGMA_N)).astype(int)
    return p_[sigma, x]


def sample(sigma):
    seed = 0
    key = jax.random.PRNGKey(0)
    print(sigma)
    out = sigma * jax.random.normal(key, sigma.shape)
    out = (out + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return out


score_norm_ = score(
    sample(sigma[None].repeat(10000, 0).flatten()),
    sigma[None].repeat(10000, 0).flatten()
).reshape(10000, -1)
score_norm_ = (score_norm_ ** 2).mean(0)


def score_norm(sigma):
    sigma = jnp.log(sigma / jnp.pi)
    sigma = (sigma - jnp.log(SIGMA_MIN)) / (jnp.log(SIGMA_MAX) - jnp.log(SIGMA_MIN)) * SIGMA_N
    sigma = jnp.round(jnp.clip(sigma, 0, SIGMA_N)).astype(int)
    return score_norm_[sigma]

print(jnp.pi)