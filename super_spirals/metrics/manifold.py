import tensorflow as tf
from .pairwise import minkowski_distances
from ..optimize import binary_search


@tf.function
def double_standardized_squared_minkowski_distances(X, Y=None, p=2):
    D_squared = tf.square(minkowski_distances(X, Y, p=p))

    shape = tf.shape(D_squared)

    I = tf.eye(shape[0])

    J = I - (1 / tf.reduce_sum(I)) * tf.ones(shape)

    return -0.5 * J @ D_squared @ J


@tf.function
def stress_loss(y_true, y_pred, p_true=2, p_pred=2):

    d_true = minkowski_distances(y_true, p=p_true)

    d_pred = minkowski_distances(y_pred, p=p_pred)

    numerator = tf.reduce_sum(tf.square(d_true - d_pred))
    denominator = tf.reduce_sum(tf.square(d_true))

    return tf.sqrt(numerator / denominator)


@tf.function
def strain_loss(y_true, y_pred, p_true=2, p_pred=2):

    b_true = double_standardized_squared_minkowski_distances(y_true, p=p_true)

    b_pred = double_standardized_squared_minkowski_distances(y_pred, p=p_pred)

    numerator = tf.reduce_sum(tf.square(b_true - b_pred))
    denominator = tf.reduce_sum(tf.square(b_true))

    return tf.sqrt(numerator / denominator)


@tf.function
def lle_loss(y_true, y_pred=None, neighours=10, minkowski=2, l2=0.1):
    dtype = y_true.dtype
    d_true = minkowski_distances(y_true, p=minkowski)
    d_true_diag_na = tf.negative(d_true)
    neighbour_distance, neighbour_index = tf.nn.top_k(d_true_diag_na, k=neighours + 1)

    loss = tf.dtypes.cast(0.0, dtype)
    length = tf.shape(y_true)[0]
    features = tf.shape(y_true)[1]
    W = tf.zeros((neighours, 1))
    for index in range(length):
        y = tf.reshape(tf.transpose(tf.gather(y_true, index)), (-1, 1))
        X = tf.transpose(tf.gather(y_true, neighbour_index[index, 1:]))

        W = tf.linalg.lstsq(X, y, l2_regularizer=l2, fast=True)

        # this is not true constrained least-squares,
        #  but is rather based of the scikit implementation
        # TODO get rid of softmax
        W_n = (W) / tf.reduce_sum(W)

        y = tf.reshape(tf.transpose(tf.gather(y_pred, index)), (-1, 1))
        X = tf.transpose(tf.gather(y_pred, neighbour_index[index, 1:]))

        loss += tf.reduce_sum(tf.square(y - X @ W_n))

    return loss / (tf.dtypes.cast(length, dtype) * tf.dtypes.cast(features, dtype))


@tf.function(experimental_relax_shapes=True)
def p_conditional(d_true, sigma):

    p = tf.math.divide_no_nan(-tf.square(d_true), (2 * sigma))

    return tf.math.softmax(p, axis=1)


def test__p_conditional():
    D = tf.abs(tf.random.normal((40, 40), 0, 1)) + 0.01
    p = p_conditional(D, 2)

    tf.debugging.assert_near(tf.reduce_sum(p, axis=1), tf.ones((tf.shape(p)[0],)), 0)


@tf.function(experimental_relax_shapes=True)
def p_perplexity(d_true, sigma):
    p_true_conditional = p_conditional(d_true, sigma)

    entropy = -1 * tf.math.divide_no_nan(
        p_true_conditional * (tf.math.log(p_true_conditional)), tf.math.log(2.0)
    )

    return 2 ** tf.math.reduce_sum(entropy, axis=1)


@tf.function(experimental_relax_shapes=True)
def find_optimal_sigmas(d_true, preplexity):
    """For each row of distances matrix, find sigma that results
    in target perplexity for that role."""

    sigmas = binary_search(lambda sigma: p_perplexity(d_true, sigma), preplexity)
    return sigmas


@tf.function(experimental_relax_shapes=True)
def tsne_loss(y_true, y_pred, p_true=2, p_pred=2, perpexity=1.0):
    length = tf.shape(y_true)[0]

    null_diag = tf.dtypes.cast(
        tf.linalg.set_diag(tf.ones((length, length)), tf.zeros((length))), y_true.dtype
    )

    # High dimensional space
    d_true = minkowski_distances(y_true, p=p_true)

    all_perplexities = tf.ones((length)) * perpexity

    sigma = find_optimal_sigmas(d_true, all_perplexities)  # tf.stop_gradient

    p_true_conditional = p_conditional(d_true, sigma)
    p_true = (p_true_conditional + tf.transpose(p_true_conditional)) / (
        2 * tf.reduce_sum(null_diag, axis=0)
    )

    # Low dimensional space
    d_pred = minkowski_distances(y_pred, p=p_pred)
    q_pred_numerator = (1 + d_pred ** 2) ** -1
    q_pred_denominator = tf.reduce_sum(q_pred_numerator * null_diag, axis=0)
    q_pred = q_pred_numerator / q_pred_denominator

    # KL divergence
    p_true_null = (p_true * null_diag) + 1e-8
    q_pred_null = (q_pred * null_diag) + 1e-8

    return tf.reduce_mean(
        p_true_null * tf.math.log(tf.math.divide_no_nan(p_true_null, q_pred_null))
    )
