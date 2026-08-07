from functools import partial

import jax
import jax.numpy as jnp

from gcsbm.csbm import (
    NULL_LABEL,
    CSBMParam,
    CSBMParamPrior,
    csbm_log_likelihood,
    ctx_log_prob,
)


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
            label == NULL_LABEL, resample, lambda: (label, key)
        )

        return (key, node + 1, score), new_label

    # We iteratively re-sample labels and update the scores accordingly
    _, labels = jax.lax.scan(
        update_label, init=(key, jnp.array(0), score), length=len(features)
    )
    return labels


def resample_theta(
    key: jax.Array,
    labels: jax.Array,
    theta: CSBMParam,
    adj: jax.Array,
    features: jax.Array,
    theta_prior: CSBMParamPrior,
):
    # theta is composed of three parameters:
    # label probabilities, intra/inter community connectivity probabilities,
    # and label-conditioned feature averages
    kl, kc, kf = jax.random.split(key, 3)

    # we use a Dirichlet and beta priors for the former, and a Gaussian for the latter

    # A posterior distribution over labels
    label_prior = theta_prior.label_concentration
    label_count = jnp.zeros_like(label_prior).at[labels].add(1)
    label_posterior = label_count + label_prior

    label_dist = jax.random.dirichlet(kl, label_posterior)

    # A posterior distribution over intra- and inter-label connectivity
    conn_prior = theta_prior.conn_concentration
    label_i, label_j = jnp.meshgrid(labels, labels)
    conn_count = jnp.zeros_like(conn_prior).at[label_i, label_j, 0].add(adj)
    conn_count = conn_count.at[label_i, label_j, 1].add(1 - adj)

    # Divide by 2 since every edge is counted twice
    conn_posterior = conn_count / 2 + conn_prior
    i, j = jnp.triu_indices(len(conn_posterior))
    conn_samples = jax.random.beta(
        kc, conn_posterior[i, j, 0], conn_posterior[i, j, 1]
    )
    conn_dist = theta.conn_dist.at[i, j].set(conn_samples)
    conn_dist = conn_dist.at[j, i].set(conn_samples)

    # A posterior distribution over label-wise features
    features_posterior_mean = jnp.zeros_like(theta_prior.mu_mean)
    features_posterior_mean = features_posterior_mean.at[labels].add(features)
    features_posterior_mean = features_posterior_mean / label_count

    sigma_sq = theta.sigma**2
    sigma_prior_sq = theta_prior.sigma_mean**2

    features_posterior_mean = (
        label_count[:, None] * sigma_prior_sq * features_posterior_mean
        + sigma_sq * theta_prior.mu_mean
    ) / (label_count[:, None] * sigma_prior_sq + sigma_sq)
    features_posterior_std = jnp.sqrt(
        (sigma_sq * sigma_prior_sq)
        / (label_count[:, None] * sigma_prior_sq + sigma_sq)
    )

    mu_dist = features_posterior_mean + features_posterior_std[
        :, None
    ] * jax.random.normal(kf, shape=features_posterior_mean.shape)

    return theta.relace(
        label_dist=label_dist, conn_dist=conn_dist, mu_dist=mu_dist
    )


def step(
    carry: tuple[jax.Array, jax.Array, jax.Array],
    _,
    adj: jax.Array,
    features: jax.Array,
    theta_prior: CSBMParamPrior,
):
    key, labels, theta = carry

    nk, kl, kt = jax.random.split(key, 3)

    # We first update labels
    labels = resample_labels(kl, labels, theta, adj, features)

    # We then update theta
    ntheta = resample_theta(kt, labels, theta, adj, features, theta_prior)

    # Return carry and scanned output
    return (nk, labels, ntheta), (labels, ntheta)


def sample(
    labels: jax.Array,
    theta: CSBMParam,
    adj: jax.Array,
    features: jax.Array,
    theta_prior: CSBMParamPrior,
    steps: int = 100,
    seed: int = 42,
):
    key = jax.random.key(seed)

    _, (labels, thetas) = jax.lax.scan(
        f=partial(step, adj=adj, features=features, theta_prior=theta_prior),
        init=(key, labels, theta),
        length=steps,
    )

    return labels, thetas
