import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Int


def debug_shape(x):
    import equinox as eqx

    return eqx.tree_pprint(jax.tree.map(lambda x: {x.shape: x.dtype}, x))


def leaky_hard_sigmoid(x, alpha=0.01):
    return jnp.maximum(jnp.minimum(1.0 + alpha * x, (x + 1) / 2), alpha * x)


def leaky_hard_tanh(x, alpha=0.01):
    return jnp.maximum(jnp.minimum(1.0 + alpha * x, x), alpha * x)


def transformer_positional_encoding(
    d_model: int, time_index: Int[Array, ""]
) -> jnp.ndarray:
    """
    Generate a positional encoding vector for a given time index.

    Args:
        time_index (int): The time step index to encode.
        d_model (int): The dimensionality of the encoding vector.

    Returns:
        jnp.ndarray: A positional encoding vector of shape (d_model,).
    """
    position = time_index
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))
    pos_encoding = jnp.zeros(d_model)
    pos_encoding = pos_encoding.at[0::2].set(jnp.sin(position * div_term))
    pos_encoding = pos_encoding.at[1::2].set(jnp.cos(position * div_term))
    return pos_encoding


def gram_schmidt(vectors):
    """Implementation of the modified Gram-Schmidt orthonormalization algorithm.

    We assume here that the vectors are linearly independent. Zero vectors will be
    left unchanged, but will also consume an iteration against `num_vectors`.

    From [1]: "MGS is numerically equivalent to Householder QR factorization
    applied to the matrix A augmented with a square matrix of zero elements on
    top."

    Historical note, see [1]: "modified" Gram-Schmidt was derived by Laplace [2],
    for elimination and not as an orthogonalization algorithm. "Classical"
    Gram-Schmidt actually came later [2]. Classical Gram-Schmidt has a sometimes
    catastrophic loss of orthogonality for badly conditioned matrices, which is
    discussed further in [1].

    #### References

    [1] Bjorck, A. (1994). Numerics of gram-schmidt orthogonalization. Linear
        Algebra and Its Applications, 197, 297-316.

    [2] P. S. Laplace, Thiorie Analytique des Probabilites. Premier Supple'ment,
        Mme. Courtier, Paris, 1816.

    [3] E. Schmidt, über die Auflosung linearer Gleichungen mit unendlich vielen
        Unbekannten, Rend. Circ. Mat. Pulermo (1) 25:53-77 (1908).

    Args:
      vectors: A Tensor of shape `[d, n]` of `d`-dim column vectors to
        orthonormalize.

    Returns:
      A Tensor of shape `[d, n]` corresponding to the orthonormalization.
    """
    num_vectors = vectors.shape[-1]

    def body_fn(vecs, i):
        # Slice out the vector w.r.t. which we're orthogonalizing the rest.
        u = jnp.nan_to_num(vecs[:, i] / jnp.linalg.norm(vecs[:, i]))
        # Find weights by dotting the d x 1 against the d x n.
        weights = u @ vecs
        # Project out vector `u` from the trailing vectors.
        masked_weights = jnp.where(jnp.arange(num_vectors) > i, weights, 0.0)
        vecs = vecs - jnp.outer(u, masked_weights)
        return vecs, None

    vectors, _ = lax.scan(body_fn, vectors, jnp.arange(num_vectors - 1))
    vec_norm = jnp.linalg.norm(vectors, axis=0, keepdims=True)
    return jnp.nan_to_num(vectors / vec_norm)
