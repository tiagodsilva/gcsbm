from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from gcsbm.csbm import NULL_LABEL, CSBMParam, csbm_log_likelihood, ctx_log_prob


def resample_labels(
    key: jax.Array,
    labels: jax.Array,
    theta: CSBMParam,
    adj: jax.Array,
    features: jax.Array,
):
    # Compute the initial score
    score = csbm_log_likelihood(adj, features, labels, theta)

    def update_score(
        node: jax.Array,
        score: jax.Array,
        old_label: jax.Array,
        new_label: jax.Array,
    ):
        # We do this incrementally to avoid repeated redundant computations
        label_log_prob = jnp.log(theta.label_dist)
        conn_log_prob = jnp.log(theta.conn_dist)

        # There are three components: label probabilities, edge probabilities, and feature probabilities
        score_shift = label_log_prob[new_label] - label_log_prob[old_label]
        score_shift = score_shift + (
            conn_log_prob[new_label, labels] - conn_log_prob[old_label, labels]
        )
        score_shift = (
            score_shift
            + ctx_log_prob(features[node], new_label, theta)
            - ctx_log_prob(features[node], old_label, theta)
        )
        return score + score_shift

    def update_label(carry: jax.Array, label: jax.Array):
        key, node, score = carry

        # Compute scores for each new label
        def resample():
            scores = jax.vmap(update_score, in_axes=(None, None, None, 0))(
                node, score, label, jnp.arange(len(theta.label_dist))
            )
            logits = jax.nn.log_softmax(scores)

            newkey, subkey = jax.random.split(key, 2)
            new_label = jax.random.categorical(subkey, logits)

            return new_label, newkey

        new_label, key = jax.lax.cond(
            label == NULL_LABEL, lambda: (label, key), resample
        )

        return (key, node + 1, score), new_label

    (key, _, score), labels = jax.lax.scan(
        update_label, init=(key, jnp.array(0), score), length=len(features)
    )

    # We iteratively re-sample labels and update the scores accordingly
    _, labels = jax.lax.scan(
        update_label, init=(key, jnp.array(0), score), length=len(features)
    )
    return labels


def resample_theta(key: jax.Array, theta: CSBMParam):
    return


def step(
    carry: tuple[jax.Array, jax.Array, jax.Array],
    _,
    adj: jax.Array,
    features: jax.Array,
    K: int,
):
    key, labels, theta = carry

    nk, kl, kt = jax.random.split(key, 3)

    # We first update labels
    labels = resample_labels(kl, labels, theta, adj, features)

    # We then update theta
    ntheta = sample_from_posterior(kt, theta)

    # Return carry and scanned output
    return (nk, labels, ntheta), (labels, ntheta)


def sample(
    labels: jax.Array,
    theta: CSBMParam,
    adj: jax.Array,
    features: jax.Array,
    num_classes: int,
    steps: int = 100,
    seed: int = 42,
):
    key = jax.random.key(seed)

    _, (labels, thetas) = jax.lax.scan(
        f=partial(step, adj=adj, features=features, K=num_classes),
        init=(key, labels, theta),
        length=steps,
    )

    return labels, thetas
