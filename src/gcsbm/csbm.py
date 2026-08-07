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


@struct.dataclass
class CSBMParamPrior:
    # We use conjugate priors for our model
    label_concentration: jax.Array  # Dirichlet prior, (K,)
    conn_concentration: jax.Array  # Beta prior, (K, K, 2)
    mu_mean: jax.Array  # Gaussian prior, (K, d)
    mu_sigma: jax.Array  # Gaussian prior, (K, 1)

    @classmethod
    def non_informative(cls, num_labels: int, num_features: int = 1):
        return cls(
            label_concentration=jnp.ones(num_labels),
            conn_concentration=jnp.ones((num_labels, num_labels, 2)),
            mu_mean=jnp.zeros((num_labels, num_features)),
            mu_sigma=jnp.ones(num_labels),
        )

    @staticmethod
    def sample_label(key: jax.Array, con: jax.Array):
        return jax.random.dirichlet(key, con)

    @staticmethod
    def sample_conn(key: jax.Array, con: jax.Array):
        i, j = jnp.triu_indices(len(con))
        num_labels = len(con)

        # Sample the connectivity matrix
        conn_samples = jax.random.beta(key, con[i, j, 0], con[i, j, 1])

        # Assign the sampled values to the connectivity matrix
        conn_dist = jnp.zeros((num_labels, num_labels))
        conn_dist = conn_dist.at[i, j].set(conn_samples)
        return conn_dist.at[j, i].set(conn_samples)

    @staticmethod
    def sample_mu(key: jax.Array, mean: jax.Array, std: jax.Array):
        return mean + std[:, None] * jax.random.normal(key, shape=mean.shape)


def ctx_log_prob(feature: jax.Array, label: jax.Array, theta: CSBMParam):
    return jsp.stats.norm.logpdf(
        feature, theta.mu_dist[label], theta.sigma
    ).sum()


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
    conn_upper = conn_matrix[i, j]
    adj_upper = adj[i, j]

    adj_log_prob = jnp.log(
        adj_upper * conn_upper + (1 - adj_upper) * (1 - conn_upper)
    ).sum()

    # Compute the log-likelihood of the labels
    labels_log_prob = jnp.log(theta.label_dist[labels]).sum()

    # Compute the log-likelihood of the features
    features_log_prob = jax.vmap(ctx_log_prob, in_axes=(0, 0, None))(
        features, labels, theta
    ).sum()

    # Compute the CSBM's log-likelihood
    return adj_log_prob + labels_log_prob + features_log_prob


def simulate(
    key: jax.Array,
    num_nodes: int,
    theta_prior: CSBMParamPrior,
    sigma: float,
    missing_rate: float = 0.5,
):
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

    # Sample labels from label dist, adjacency matrix from conn_dist,
    # and features from mu_dist
    nkm, nkl, nkc, nkf = jax.random.split(nk, 4)

    # true labels
    true_labels = jax.random.categorical(
        nkl, logits=jnp.log(label_dist), shape=(num_nodes,)
    )

    # adjacency matrix
    i, j = jnp.triu_indices(num_nodes, k=1)
    adj_flatten = jax.random.bernoulli(
        nkc, conn_dist[true_labels[i], true_labels[j]]
    )
    adj = jnp.zeros((num_nodes, num_nodes)).at[i, j].set(adj_flatten)
    adj = adj.at[j, i].set(adj_flatten)

    # features
    features = CSBMParamPrior.sample_mu(
        nkf, mu_dist[true_labels], jnp.ones_like(true_labels) * sigma
    )

    # mask labels
    missing_mask = jax.random.bernoulli(
        nkm, p=missing_rate, shape=(num_nodes,)
    )
    labels = jnp.where(missing_mask, NULL_LABEL, true_labels)

    return adj, labels, features
