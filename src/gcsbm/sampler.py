from functools import partial

import jax
import jax.numpy as jnp


def label_score(Y_U: jax.Array, Y_L: jax.Array, A: jax.Array, X: jax.Array):
    return


def sample_from_posterior(key: jax.Array, theta: jax.Array):
    return


def step(
    carry: tuple[jax.Array, jax.Array, jax.Array],
    _,
    A: jax.Array,
    X: jax.Array,
    Y_L: jax.Array,
    K: int,
):
    key, Y_U, theta = carry

    nk, kl, kt = jax.random.split(key, 3)

    # We first update Y_U
    scores = label_score(Y_U, Y_L, A, X)
    logits = jax.nn.log_softmax(scores, axis=0)
    labels = jax.random.categorical(kl, logits)

    # We then update theta
    ntheta = sample_from_posterior(kt, theta)

    # Return carry and scanned output
    return (nk, labels, ntheta), (labels, ntheta)


def sample(
    labels: jax.Array,
    theta: jax.Array,
    A: jax.Array,
    X: jax.Array,
    Y: jax.Array,
    K: int,  # Number of labels
    steps: int,
    seed: int = 42,
):
    key = jax.random.key(seed)

    _, (labels, thetas) = jax.lax.scan(
        f=partial(step, A=A, X=X, Y_L=Y, K=K),
        init=(key, labels, theta),
        length=steps,
    )

    return Y_Us, thetas
