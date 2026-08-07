__author__ = "Antigravity"

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from gcsbm.csbm import NULL_LABEL, CSBMParam, CSBMParamPrior
from gcsbm.sampler import sample


def generate_sbc_data(key, num_nodes, theta_prior, sigma, missing_rate=0.5):
    nk, kl, kc, kf = jax.random.split(key, 4)
    label_dist = CSBMParamPrior.sample_label(
        kl, theta_prior.label_concentration
    )
    conn_dist = CSBMParamPrior.sample_conn(kc, theta_prior.conn_concentration)
    mu_dist = CSBMParamPrior.sample_mu(
        kf,
        theta_prior.mu_mean,
        theta_prior.mu_sigma * jnp.ones_like(label_dist),
    )
    theta = CSBMParam(
        label_dist=label_dist,
        conn_dist=conn_dist,
        mu_dist=mu_dist,
        sigma=sigma,
    )

    nkm, nkl, nkc, nkf = jax.random.split(nk, 4)

    true_labels = jax.random.categorical(
        nkl, logits=jnp.log(label_dist), shape=(num_nodes,)
    )

    i, j = jnp.triu_indices(num_nodes, k=1)
    adj_flatten = jax.random.bernoulli(
        nkc, conn_dist[true_labels[i], true_labels[j]]
    )
    adj = jnp.zeros((num_nodes, num_nodes)).at[i, j].set(adj_flatten)
    adj = adj.at[j, i].set(adj_flatten)

    features = CSBMParamPrior.sample_mu(
        nkf, mu_dist[true_labels], jnp.ones_like(true_labels) * sigma
    )

    missing_mask = jax.random.bernoulli(
        nkm, p=missing_rate, shape=(num_nodes,)
    )
    labels = jnp.where(missing_mask, NULL_LABEL, true_labels)

    return theta, adj, labels, features, true_labels


@partial(jax.jit, static_argnames=("num_nodes", "steps", "burn_in"))
def run_sbc_trial(
    key, num_nodes, theta_prior, sigma, missing_rate, steps, burn_in
):
    # 1. Generate data from the prior
    sim_key, sample_key = jax.random.split(key)
    true_theta, adj, labels, features, _ = generate_sbc_data(
        sim_key, num_nodes, theta_prior, sigma, missing_rate
    )

    # 2. Sample from the posterior
    seed = jax.random.randint(sample_key, (), 0, 1000000)
    _, thetas = sample(
        labels=labels,
        adj=adj,
        features=features,
        theta_prior=theta_prior,
        sigma=sigma,
        steps=steps,
        seed=seed,
    )

    # Discard burn-in
    post_label_dist = thetas.label_dist[burn_in:]
    post_conn_dist = thetas.conn_dist[burn_in:]
    post_mu_dist = thetas.mu_dist[burn_in:]

    # 3. Compute permutation invariant representations to avoid label switching issues
    # sort by label probabilities
    def sort_params(label_dist, conn_dist, mu_dist):
        sort_idx = jnp.argsort(label_dist)
        label_dist = label_dist[sort_idx]
        mu_dist = mu_dist[sort_idx]

        # for conn_dist, we need to sort both dimensions
        conn_dist = conn_dist[sort_idx, :]
        conn_dist = conn_dist[:, sort_idx]
        return label_dist, conn_dist, mu_dist

    true_label_dist, true_conn_dist, true_mu_dist = sort_params(
        true_theta.label_dist, true_theta.conn_dist, true_theta.mu_dist
    )

    post_label_dist, post_conn_dist, post_mu_dist = jax.vmap(sort_params)(
        post_label_dist, post_conn_dist, post_mu_dist
    )

    # 4. Compute ranks
    rank_label_dist = jnp.sum(post_label_dist < true_label_dist, axis=0)
    rank_conn_dist = jnp.sum(post_conn_dist < true_conn_dist, axis=0)
    rank_mu_dist = jnp.sum(post_mu_dist < true_mu_dist, axis=0)

    return rank_label_dist, rank_conn_dist, rank_mu_dist


def main():
    print("Setting up SBC...")
    K = 2
    d = 4
    num_nodes = 25
    sigma = 1.0
    missing_rate = 0.3
    steps = 500
    burn_in = 100
    num_trials = 1000

    theta_prior = CSBMParamPrior.non_informative(num_labels=K, num_features=d)

    key = jax.random.key(42)
    keys = jax.random.split(key, num_trials)

    print(f"Running {num_trials} SBC trials...")

    vmapped_run_sbc = jax.vmap(
        run_sbc_trial, in_axes=(0, None, None, None, None, None, None)
    )

    ranks_label_dist, ranks_conn_dist, ranks_mu_dist = vmapped_run_sbc(
        keys,
        num_nodes,
        theta_prior,
        sigma,
        missing_rate,
        steps,
        burn_in,
    )

    ranks_label_dist = np.array(ranks_label_dist)
    ranks_conn_dist = np.array(ranks_conn_dist)
    ranks_mu_dist = np.array(ranks_mu_dist)

    print("Plotting SBC histograms...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    num_samples = steps - burn_in

    # We can flatten the ranks for similar parameters to see the overall uniformity
    axes[0].hist(ranks_label_dist.flatten(), bins=20, density=True, alpha=0.7)
    axes[0].set_title("Label Distribution Ranks")
    axes[0].axhline(1.0 / num_samples, color="r", linestyle="--")

    axes[1].hist(ranks_conn_dist.flatten(), bins=20, density=True, alpha=0.7)
    axes[1].set_title("Connectivity Matrix Ranks")
    axes[1].axhline(1.0 / num_samples, color="r", linestyle="--")

    axes[2].hist(ranks_mu_dist.flatten(), bins=20, density=True, alpha=0.7)
    axes[2].set_title("Feature Means Ranks")
    axes[2].axhline(1.0 / num_samples, color="r", linestyle="--")

    plt.tight_layout()
    plt.savefig("sbc_histograms.png")
    print("SBC histograms saved to sbc_histograms.png")


if __name__ == "__main__":
    main()
