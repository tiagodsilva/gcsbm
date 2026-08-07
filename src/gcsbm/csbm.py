import flax.struct as struct
import jax
import jax.numpy as jnp
import jax.scipy as jsp

NULL_LABEL = -1


@struct.dataclass
class CSBMParam:
    label_dist: jax.Array
    conn_dist: jax.Array
    mu_dist: jax.Array

    # We assume features are drawn from a label-conditioned isotropic
    # Gaussian with average mu_dist[label] and variance sigma
    sigma: float


class CSBMParamPrior:
    # We use conjugate priors for our model
    label_concentration: jax.Array  # Dirichlet prior, (K,)
    conn_concentration: jax.Array  # Beta prior, (K, K, 2)
    mu_mean: jax.Array  # Gaussian prior, (K, d)
    mu_sigma: jax.Array  # Gaussian prior, (K, 1)


def ctx_log_prob(feature: jax.Array, label: jax.Array, theta: CSBMParam):
    return jsp.stats.norm.logpdf(feature, theta.mu_dist[label], theta.sigma)


def csbm_log_likelihood(
    adj: jax.Array,
    features: jax.Array,
    labels: jax.Array,
    theta: CSBMParam,
):
    num_nodes = len(adj)
    # Compute the log-likelihood of the adjacency matrix
    label_i, label_j = jnp.meshgrid(labels, labels)
    conn_matrix = theta.conn_dist[label_i, label_j]

    # Select only the upper triangular matrix
    i, j = jnp.triu_indices(num_nodes, k=1)
    conn_upper = conn_matrix[j, j]
    adj_upper = adj[i, j]

    adj_log_prob = jnp.log(
        adj_upper * conn_upper + (1 - adj_upper) * (1 - conn_upper)
    ).sum()

    # Compute the log-likelihood of the labels
    labels_log_prob = jnp.log(theta.label_dist.at[labels]).sum()

    # Compute the log-likelihood of the features
    features_log_prob = jax.vmap(ctx_log_prob, in_axes=(0, 0, None))(
        features, labels, theta
    ).sum()

    # Compute the CSBM's log-likelihood
    return adj_log_prob + labels_log_prob + features_log_prob
