import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.base import BaseEstimator, RegressorMixin
from toolz.curried import pipe, map, compose_left, partial
import numpy as np
from typing import Tuple, Union, List
from abc import ABCMeta
import warnings


class RBFKernelProvider(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RBFKernelProvider, self).__init__(**kwargs)
        dtype = kwargs.get("dtype", None)

        self._amplitude = self.add_weight(
            initializer=tf.constant_initializer(0), dtype=dtype, name="amplitude"
        )

        self._length_scale = self.add_weight(
            initializer=tf.constant_initializer(0), dtype=dtype, name="length_scale"
        )

    def call(self, x):
        # Never called -- this is just a layer so it can hold variables
        # in a way Keras understands.
        return x

    @property
    def kernel(self):
        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=tf.nn.softplus(0.1 * self._amplitude),
            length_scale=tf.nn.softplus(5.0 * self._length_scale),
        )


class DeepVariationalGaussianProcess(tf.keras.Model, BaseEstimator, RegressorMixin):
    def __init__(
        self,
        kernel: callable = RBFKernelProvider(),
        num_inducing_points: int = 20,
        alpha: float = 1e-6,
        elbo_weight=1,
        hidden_layer_sizes: Tuple[int] = (),
        l1_ratio: float = 0.0,
        activation: str = "selu",
        n_iter: int = 20,
        solver: Union[str, ABCMeta] = "adam",
        batch_size: float = 100,
        shuffle: bool = True,
        learning_rate_init: float = 0.001,
        validation_fraction: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-08,
        momentum: float = 0.9,
        nesterovs_momentum: bool = True,
        early_stopping: bool = False,
        n_iter_no_change: int = 10,
        tol: float = 0.0001,
        verbose: int = 1,
        callbacks: List = [],
    ):
        super(DeepVariationalGaussianProcess, self).__init__()

        # define model
        self.kernel = kernel
        self.num_inducing_points = num_inducing_points
        self.alpha = alpha
        self.elbo_weight = elbo_weight
        self.hidden_layer_sizes = hidden_layer_sizes
        self.l1_ratio = l1_ratio
        self.activation = activation

        hidden_units = (
            *hidden_layer_sizes,
            1,
        )
        self.hidden_units = hidden_units
        self.hidden_tranformations = pipe(
            hidden_units,
            map(
                lambda d: tf.keras.layers.Dense(
                    units=d,
                    kernel_initializer="ones",
                    use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(
                        l1=alpha * l1_ratio, l2=alpha * (1 - l1_ratio)
                    ),
                    activation=activation,
                )
            ),
        )

        self.final_transformation = tfp.layers.VariationalGaussianProcess(
            kernel_provider=kernel,
            num_inducing_points=num_inducing_points,
            event_shape=[1],
            jitter=alpha,
        )

        # get optmizer
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        if solver == "auto" or solver.title() == "Adam":
            self.solver_ = tf.keras.optimizers.Adam(
                learning_rate=learning_rate_init,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
            )
        elif solver.title() == "SGD":
            self.solver_ = tf.keras.optimizers.SGD(
                learning_rate=learning_rate_init,
                momentum=momentum,
                nesterov=nesterovs_momentum,
            )
        elif type(solver) is str:
            self.solver_ = getattr(tf.keras.optimizers, solver)(
                learning_rate=learning_rate_init
            )
        elif type(solver) is ABCMeta:
            self.solver_ = solver
        else:
            warnings.warn("Not a valid optimizer, reverting to Adam")
            self.solver_ = tf.keras.optimizers.Adam(
                learning_rate=learning_rate_init,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
            )

        # get batch size
        if batch_size == None or batch_size == "auto":
            self.batch_size = 32
        else:
            self.batch_size = batch_size

        self.validation_fraction = validation_fraction
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.n_iter_no_change = n_iter_no_change
        self.early_stopping = early_stopping
        self.tol = tol
        self.callbacks = callbacks
        # Options
        if early_stopping:
            self.callbacks_ = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", min_delta=tol, patience=n_iter_no_change
                ),
                *callbacks,
            ]
        else:
            self.callbacks_ = [*callbacks]

        self.verbose = verbose

    def call(self, inputs: Union[np.ndarray, tf.Tensor]):
        return pipe(inputs, *self.hidden_tranformations, self.final_transformation)

    def fit(self, X: Union[np.ndarray, tf.Tensor], y: Union[np.ndarray, tf.Tensor]):
        loss = lambda y, rv_y: rv_y.variational_loss(
            y, kl_weight=np.array(self.batch_size, X.dtype) / X.shape[0]
        )

        self.compile(optimizer=self.solver_, loss=loss)

        # TODO: Figure out why it fails when not called before fitting
        self.call(X)

        super().fit(
            x=X.astype("float32"),
            y=y.astype("float32"),
            shuffle=self.shuffle,
            epochs=self.n_iter,
            batch_size=self.batch_size,
            validation_split=self.validation_fraction,
            callbacks=self.callbacks_,
            verbose=self.verbose,
        )

        return self

    def predict(self, X: Union[np.ndarray, tf.Tensor], return_std=False):
        posterior_dist = self.call(X.astype("float32"))

        if return_std:
            return posterior_dist.mean(), posterior_dist.stddev()
        else:
            return posterior_dist.mean()
