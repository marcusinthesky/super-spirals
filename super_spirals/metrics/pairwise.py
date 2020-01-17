import tensorflow as tf
from numpy import pi


@tf.function(experimental_relax_shapes=True)
def minkowski_distances(X: tf.Tensor, Y: tf.Tensor = None, p: int = 2):
    X_expanded = tf.expand_dims(X, -1)
    X_tiled = tf.tile(X_expanded, [1, 1, tf.shape(X_expanded)[0]])

    if Y is None:
        Y_expanded = tf.transpose(X_expanded)
    else:
        Y_expanded = tf.transpose(tf.expand_dims(Y, -1))

    return tf.norm(X_tiled - Y_expanded, ord=p, axis=1)


def test__minkowski_distances():
    data = tf.constant(
        [
            [-0.59794587, 1.084908, -0.4533812],
            [1.061402, -0.41524476, 1.1109126],
            [-0.1625053, 0.90884435, 2.1842542],
        ],
        dtype=tf.float32,
    )

    expected = tf.constant(
        [
            [0.0, 2.7296352, 2.679128],
            [2.7296352, 0.0, 2.0983858],
            [2.679128, 2.0983858, 0.0],
        ],
        dtype=tf.float32,
    )

    tf.debugging.assert_near(s(data), expected)


@tf.function(experimental_relax_shapes=True)
def cosine_similarity(X: tf.Tensor, Y: tf.Tensor = None):

    if Y is None:
        Y = X

    X_norm = tf.math.l2_normalize(X, axis=1)
    Y_norm = tf.math.l2_normalize(Y, axis=1)

    return tf.linalg.matmul(X_norm, Y_norm, adjoint_b=True)


def test__cosine_similarity():
    from sklearn.metrics.pairwise import cosine_similarity as scikit_cosine_similarity

    data = tf.constant(
        [
            [-0.59794587, 1.084908, -0.4533812],
            [1.061402, -0.41524476, 1.1109126],
            [-0.1625053, 0.90884435, 2.1842542],
        ],
        dtype=tf.float32,
    )

    expected = tf.constant(scikit_cosine_similarity(data.numpy()))

    tf.debugging.assert_near(cosine_similarity(data), expected)


@tf.function(experimental_relax_shapes=True)
def cosine_distances(X: tf.Tensor, Y: tf.Tensor = None):

    d = cosine_similarity(X, Y)

    # NOTE: it seems the scikit-implementation is slightly odd in that it just adds and clips
    # return 2. *tf.math.acos(d) / tf.constant(pi)

    d *= -1.0
    d += 1.0

    return tf.clip_by_value(d, 0, 2)


def test__cosine_distances():
    from sklearn.metrics.pairwise import cosine_distances as scikit_cosine_distances

    data = tf.constant(
        [
            [-0.59794587, 1.084908, -0.4533812],
            [1.061402, -0.41524476, 1.1109126],
            [-0.1625053, 0.90884435, 2.1842542],
        ],
        dtype=tf.float32,
    )

    expected = tf.constant(scikit_cosine_distances(data.numpy()))

    tf.debugging.assert_near(cosine_distances(data), expected)
