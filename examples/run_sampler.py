import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from gcsbm.csbm import CSBMParamPrior, simulate
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

    print("Simulating graph with 30% missing data...")
    adj, labels, features = simulate(
        sim_key, num_nodes, theta_prior, sigma, missing_rate=0.3
    )

    missing_count = jnp.sum(labels < 0)
    print(f"Total nodes: {num_nodes}")
    print(
        f"Missing labels: {missing_count} ({(missing_count / num_nodes) * 100:.1f}%)"
    )

    print("Running sampler for 100 steps...")
    sampled_labels, _ = sample(
        labels=labels,
        adj=adj,
        features=features,
        theta_prior=theta_prior,
        sigma=sigma,
        steps=100,
        seed=42,
    )

    print("Sampling complete.")
    print("Final sampled labels shape:", sampled_labels.shape)

    # Display per node probabilites of label = 1
    counts = sampled_labels.sum(axis=0)
    plt.figure(figsize=(10, 4))
    plt.bar(range(num_nodes), counts)
    plt.savefig("node_probabilities.png")


if __name__ == "__main__":
    main()
