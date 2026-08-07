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
    is_missing: jax.Array,
):
    # Compute the initial score
    score = csbm_log_likelihood(adj, features, labels, theta)

    def update_score(
        node: jax.Array,
        score: jax.Array,
        old_label: jax.Array,
        new_label: jax.Array,
        current_labels: jax.Array,
    ):
        # We do this incrementally to avoid repeated redundant computations
        label_log_prob = jnp.log(theta.label_dist)

        conn_prob_new = theta.conn_dist[new_label, current_labels]
        conn_prob_old = theta.conn_dist[old_label, current_labels]

        edge_log_prob_new = jnp.log(
            adj[node] * conn_prob_new + (1 - adj[node]) * (1 - conn_prob_new)
        )
        edge_log_prob_old = jnp.log(
            adj[node] * conn_prob_old + (1 - adj[node]) * (1 - conn_prob_old)
        )

        # # Ignore self loop since the graph has no self loops
        edge_log_prob_new = edge_log_prob_new.at[node].set(0.0)
        edge_log_prob_old = edge_log_prob_old.at[node].set(0.0)

        # There are three components: label probabilities, edge probabilities, and feature probabilities
        score_shift = label_log_prob[new_label] - label_log_prob[old_label]
        score_shift = (
            score_shift + edge_log_prob_new.sum() - edge_log_prob_old.sum()
        )

        score_shift = (
            score_shift
            + ctx_log_prob(features[node], new_label, theta)
            - ctx_log_prob(features[node], old_label, theta)
        )
        return score + score_shift

    def update_label(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        missing: jax.Array,
    ):
        key, node, score, current_labels = carry
        label = current_labels[node]

        # Compute scores for each new label
        def resample():
            scores = jax.vmap(
                update_score, in_axes=(None, None, None, 0, None)
            )(
                node,
                score,
                label,
                jnp.arange(len(theta.label_dist)),
                current_labels,
            )
            logits = jax.nn.log_softmax(scores)

            newkey, subkey = jax.random.split(key, 2)
            new_label = jax.random.categorical(subkey, logits)
            new_score = scores[new_label]

            return new_label, newkey, new_score

        new_label, key, score = jax.lax.cond(
            missing, resample, lambda: (label, key, score)
        )
        current_labels = current_labels.at[node].set(new_label)

        return (key, node + 1, score, current_labels), None

    # We iteratively re-sample labels and update the scores accordingly
    (_, _, _, labels), _ = jax.lax.scan(
        update_label,
        init=(key, jnp.array(0, dtype=jnp.int32), score, labels),
        xs=is_missing,
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
    def sample_label_con(key: jax.Array):
        label_prior = theta_prior.label_concentration
        label_count = jnp.zeros_like(label_prior).at[labels].add(1)
        label_posterior = label_count + label_prior
        return label_count, CSBMParamPrior.sample_label(key, label_posterior)

    label_count, label_dist = sample_label_con(kl)

    # A posterior distribution over intra- and inter-label connectivity
    def sample_conn_con(key: jax.Array):
        conn_prior = theta_prior.conn_concentration
        label_i, label_j = jnp.meshgrid(labels, labels)

        # Mask out diagonal since the graph has no self loops
        mask = 1.0 - jnp.eye(len(adj))

        conn_count = jnp.zeros_like(conn_prior)
        conn_count = conn_count.at[label_i, label_j, 0].add(adj * mask)
        conn_count = conn_count.at[label_i, label_j, 1].add((1 - adj) * mask)

        # Nodes (i, j) with label_i = label_j are counted twice.
        K = len(conn_prior)
        Ks = jnp.arange(K)
        conn_count = conn_count.at[Ks, Ks].set(conn_count[Ks, Ks] / 2.0)

        conn_posterior = conn_count + conn_prior
        return CSBMParamPrior.sample_conn(key, conn_posterior)

    conn_dist = sample_conn_con(kc)

    # A posterior distribution over label-wise features
    def sample_features_mean(key: jax.Array):
        features_sum = jnp.zeros_like(theta_prior.mu_mean)
        features_sum = features_sum.at[labels].add(
            features
        )

        sigma_sq = theta.sigma**2
        sigma_prior_sq = theta_prior.mu_sigma**2

        features_posterior_mean = (
            sigma_prior_sq[:, None] * features_sum
            + sigma_sq * theta_prior.mu_mean
        ) / (label_count[:, None] * sigma_prior_sq[:, None] + sigma_sq)

        features_posterior_std = jnp.sqrt(
            (sigma_sq * sigma_prior_sq)
            / (label_count * sigma_prior_sq + sigma_sq)
        )

        return CSBMParamPrior.sample_mu(
            kf, features_posterior_mean, features_posterior_std
        )

    mu_dist = sample_features_mean(kf)

    return theta.replace(
        label_dist=label_dist, conn_dist=conn_dist, mu_dist=mu_dist
    )


@jax.jit
def step(
    carry: tuple[jax.Array, jax.Array, CSBMParam],
    _,
    adj: jax.Array,
    features: jax.Array,
    theta_prior: CSBMParamPrior,
    is_missing: jax.Array,
):
    key, labels, theta = carry

    nk, kl, kt = jax.random.split(key, 3)

    # We first update labels
    nlabels = resample_labels(kl, labels, theta, adj, features, is_missing)

    # We then update theta
    ntheta = resample_theta(kt, labels, theta, adj, features, theta_prior)

    # Return carry and scanned output
    return (nk, nlabels, ntheta), (nlabels, ntheta)


def sample(
    labels: jax.Array,
    adj: jax.Array,
    features: jax.Array,
    theta_prior: CSBMParamPrior,
    sigma: float,
    steps: int = 100,
    seed: int = 42,
):
    key = jax.random.key(seed)
    key, init_key, lk, ck, mk = jax.random.split(key, 5)

    is_missing = labels == NULL_LABEL
    # Give random labels to unobserved nodes
    random_labels = jax.random.randint(
        init_key, labels.shape, 0, len(theta_prior.label_concentration)
    )
    labels = jnp.where(is_missing, random_labels, labels)

    # Initialize theta
    label_dist = CSBMParamPrior.sample_label(
        lk, theta_prior.label_concentration
    )
    conn_dist = CSBMParamPrior.sample_conn(ck, theta_prior.conn_concentration)
    mu_dist = CSBMParamPrior.sample_mu(
        mk, theta_prior.mu_mean, theta_prior.mu_sigma
    )

    theta = CSBMParam(
        label_dist=label_dist,
        conn_dist=conn_dist,
        mu_dist=mu_dist,
        sigma=sigma,
    )

    # Sample from the posterior
    _, (sampled_labels, thetas) = jax.lax.scan(
        f=partial(
            step,
            adj=adj,
            features=features,
            theta_prior=theta_prior,
            is_missing=is_missing,
        ),
        init=(key, labels, theta),
        length=steps,
    )

    return sampled_labels, thetas
