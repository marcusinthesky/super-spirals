import tensorflow as tf


@tf.function
def pinv(X: tf.Tensor) -> tf.Tensor:
    s, u, v = tf.linalg.svd(X, full_matrices=True)

    return v @ tf.linalg.diag(1 / s) @ tf.transpose(u)


def test__pinv():
    D = tf.random.normal((5, 5))
    tf.debugging.assert_near(
        tf.reduce_sum(pinv(D) @ D), tf.reduce_sum(tf.eye(5)), rtol=1e-8
    )
