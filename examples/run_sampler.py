import jax
import jax.numpy as jnp

from gcsbm.csbm import NULL_LABEL, CSBMParam, CSBMParamPrior, simulate
from gcsbm.sampler import sample


def main():
    print("Setting up priors...")
    K = 2
    d = 3
    num_nodes = 100
    sigma = 1.0

    theta_prior = CSBMParamPrior.non_informative(num_labels=K, num_features=d)

    key = jax.random.key(42)
    key, sim_key = jax.random.split(key)

    print("Simulating graph...")
    adj, true_labels, features = simulate(
        sim_key, num_nodes, theta_prior, sigma
    )

    # 30% missing data
    print("Applying 30% missing data mask...")
    key, mask_key = jax.random.split(key)
    missing_mask = jax.random.bernoulli(mask_key, p=0.3, shape=(num_nodes,))

    # Apply missing mask
    labels = jnp.where(missing_mask, NULL_LABEL, true_labels)

    missing_count = jnp.sum(labels == NULL_LABEL)
    print(f"Total nodes: {num_nodes}")
    print(
        f"Missing labels: {missing_count} ({(missing_count / num_nodes) * 100:.1f}%)"
    )

    print("Running sampler for 100 steps...")
    sampled_labels, sampled_thetas = sample(
        labels=labels,
        adj=adj,
        features=features,
        theta_prior=theta_prior,
        sigma=sigma,
        steps=100,
        seed=42
    )

    print("Sampling complete.")
    print("Final sampled labels shape:", sampled_labels.shape)


if __name__ == "__main__":
    main()
