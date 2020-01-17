import tensorflow as tf
from super_spirals.metrics.pairwise import minkowski_distances
from super_spirals.metrics.manifold import (
    p_conditional as negative_exponentiated_quadratic,
)


def _make_mmd_divergence_fn(
    distribution_b,
    test_points_reduce_axis=(),  # `None` == "all"; () == "none".
    test_points_fn=tf.convert_to_tensor,
    weight=1.0,
):
    """Creates a callable computing `MMD[a,b]` from `a`, a `tfd.Distribution`."""

    def mmd_divergence_fn(distribution_a, distribution_b):

        a_samples = test_points_fn(distribution_a)
        b_samples = distribution_b.sample(1)

        x_kernel = negative_exponentiated_quadratic(
            minkowski_distances(a_samples, a_samples), 1
        )
        y_kernel = negative_exponentiated_quadratic(
            minkowski_distances(b_samples, b_samples), 1
        )
        xy_kernel = negative_exponentiated_quadratic(
            minkowski_distances(a_samples, b_samples), 1
        )

        return (
            tf.reduce_mean(x_kernel)
            + tf.reduce_mean(y_kernel)
            - 2 * tf.reduce_mean(xy_kernel)
        )

    # Closure over: distribution_b, mmd_divergence_fn, weight.
    def _fn(distribution_a):
        """Closure that computes MMDDiv as a function of `a` as in `MMD[a, b]`."""
        with tf.name_scope("mmddivergence_loss"):
            distribution_b_ = (
                distribution_b() if callable(distribution_b) else distribution_b
            )
            mmd = mmd_divergence_fn(distribution_a, distribution_b_)
            if weight is not None:
                mmd = tf.cast(weight, dtype=mmd.dtype) * mmd

            return tf.reduce_sum(input_tensor=mmd, name="batch_total_mmd_divergence")

    return _fn


class MMDDivergenceRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(
        self,
        distribution_b,
        use_exact_kl=False,
        test_points_reduce_axis=(),  # `None` == "all"; () == "none".
        test_points_fn=tf.convert_to_tensor,
        weight=None,
    ):

        """Initialize the `MMDDivergenceRegularizer` regularizer.
        """
        super(MMDDivergenceRegularizer, self).__init__()
        self._mmd_divergence_fn = _make_mmd_divergence_fn(
            distribution_b,
            test_points_reduce_axis=test_points_reduce_axis,
            test_points_fn=test_points_fn,
            weight=weight,
        )

    def __call__(self, distribution_a):
        if hasattr(distribution_a, "_tfp_distribution"):
            distribution_a = distribution_a._tfp_distribution
        return self._mmd_divergence_fn(distribution_a)
