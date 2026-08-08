import jax
import jax.numpy as jnp

from gcsbm.csbm import CSBMParam, CSBMParamPrior, simulate
from gcsbm.sampler import sample


def mse_theta(sampled_thetas: CSBMParam, true_theta: CSBMParam):
    mse_label_dist = jnp.mean(
        jnp.sum(
            (sampled_thetas.label_dist - true_theta.label_dist[None]) ** 2,
            axis=1,
        ),
        axis=0,
    )
    mse_mu_dist = jnp.mean(
        jnp.sum(
            (sampled_thetas.mu_dist - true_theta.mu_dist[None]) ** 2,
            axis=(1, 2),
        ),
        axis=0,
    )
    mse_conn_dist = jnp.mean(
        jnp.sum(
            (sampled_thetas.conn_dist - true_theta.conn_dist[None]) ** 2,
            axis=(1, 2),
        ),
        axis=0,
    )
    return mse_label_dist, mse_mu_dist, mse_conn_dist


def main():
    print("Setting up priors...")
    K = 2
    d = 3
    num_nodes = 200
    sigma = 1.0
    steps = 4000
    missing_rate = 0.8

    theta_prior = CSBMParamPrior.non_informative(num_labels=K, num_features=d)

    key = jax.random.key(21)
    key, sim_key = jax.random.split(key)

    (adj, labels, features), (true_labels, true_theta, is_missing) = simulate(
        sim_key, num_nodes, theta_prior, sigma, missing_rate=missing_rate
    )

    print(
        f"Missing labels: {jnp.sum(is_missing)} ({(jnp.sum(is_missing) / num_nodes) * 100:.1f}%)"
    )

    print("Running the model...")
    sampled_labels, sampled_thetas = sample(
        labels=labels,
        adj=adj,
        features=features,
        theta_prior=theta_prior,
        sigma=sigma,
        steps=steps,
        seed=42,
        burnin=200,
    )

    # Compute the accuracy
    accuracy = jnp.mean(
        (sampled_labels[:, is_missing].mean(axis=0) > 0.5)
        == true_labels[is_missing],
    )
    print(f"Accuracy: {accuracy:.2f}")

    # Compute the MSE for theta
    mse_label_dist, mse_mu_dist, mse_conn_dist = mse_theta(
        sampled_thetas, true_theta
    )
    print(mse_label_dist, mse_mu_dist, mse_conn_dist)


if __name__ == "__main__":
    main()
