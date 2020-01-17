from sklearn.base import BaseEstimator, TransformerMixin
from super_spirals.layers import MMDDivergenceRegularizer
import numpy as np
from toolz.curried import pipe, map, compose_left, partial
from typing import Union, Tuple, List
import tensorflow as tf
import tensorflow_probability as tfp
import warnings
from abc import ABCMeta
from pathlib import Path


class VAE(tf.keras.Model, BaseEstimator, TransformerMixin):
    def __init__(
        self,
        reconstruction_loss: callable = "negative_log_likelihood",
        divergence_loss: str = "kl",
        elbo_weight: float = 1.0,
        hidden_layer_sizes: Tuple[int] = (4, 2),
        alpha: float = 0.01,
        l1_ratio: float = 0.0,
        activation: str = "selu",
        n_iter: int = 20,
        solver: Union[str, ABCMeta] = "adam",
        batch_size: int = 100,
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
        weights_path: Union[str, Path, None] = None,
    ):
        super(VAE, self).__init__()

        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.activation = activation
        # define model
        *hidden_units, latent_unit = hidden_layer_sizes
        self.encoder_hidden_tranformations = pipe(
            hidden_units,
            map(
                lambda d: tf.keras.layers.Dense(
                    units=d,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(
                        l1=l1_ratio * alpha, l2=(1 - l1_ratio) * alpha
                    ),
                    activation=activation,
                )
            ),
            list,
        )
        self.latent_layer = tf.keras.layers.Dense(
            tfp.layers.IndependentNormal.params_size(latent_unit),
            activation="linear",
            name="z_dist",
        )

        prior = tfp.distributions.Independent(
            tfp.distributions.Normal(loc=tf.zeros(latent_unit), scale=1),
            reinterpreted_batch_ndims=1,
        )

        self.divergence_loss = divergence_loss
        self.elbo_weight = elbo_weight

        # setup divergence loss regularizer
        if divergence_loss is "kl":
            divergence_regularizer = tfp.layers.KLDivergenceRegularizer
        elif divergence_loss is "mmd":
            divergence_regularizer = MMDDivergenceRegularizer

        self.probability_layer = tfp.layers.IndependentNormal(
            latent_unit,
            activity_regularizer=divergence_regularizer(prior, weight=elbo_weight),
        )

        self.decoder_hidden_tranformations = pipe(
            reversed(hidden_units),
            map(
                lambda d: tf.keras.layers.Dense(
                    units=d,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(
                        l1=l1_ratio * alpha, l2=(1 - l1_ratio) * alpha
                    ),
                    activation=activation,
                )
            ),
            list,
        )

        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        # get optmizer
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

        self.batch_size = batch_size
        # get batch size
        if batch_size == None or batch_size == "auto":
            self.batch_size_ = 32
        else:
            self.batch_size_ = batch_size

        self.validation_fraction = validation_fraction
        self.n_iter = n_iter
        self.shuffle = shuffle

        # Options
        self.n_iter_no_change = n_iter_no_change
        self.early_stopping = early_stopping
        self.tol = tol
        self.callbacks = callbacks
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

        # get loss
        self.reconstruction_loss = reconstruction_loss
        if reconstruction_loss is "negative_log_likelihood":
            self.reconstruction_loss_ = lambda x, rv_x: -rv_x.log_prob(x)
        else:
            self.reconstruction_loss_ = lambda x, rv_x: tf.keras.losses.mse(
                x, rv_x.mean()
            )

        self.weights_path = weights_path

    def call(self, inputs: Union[np.ndarray, tf.Tensor]):
        return pipe(
            inputs,
            *self.encoder_hidden_tranformations,
            self.latent_layer,
            self.probability_layer,
            *self.decoder_hidden_tranformations,
            self.decoder_reconstruction_transformation,
            self.decoder_reconstruction_probability
        )

    def fit(self, X: Union[np.ndarray, tf.Tensor], y=None):
        length, reconstruction_unit = X.shape
        self.decoder_reconstruction_transformation = tf.keras.layers.Dense(
            tfp.layers.IndependentNormal.params_size(reconstruction_unit),
            activation="linear",
        )
        self.decoder_reconstruction_probability = tfp.layers.IndependentNormal(
            reconstruction_unit
        )

        self.compile(optimizer=self.solver_, loss=self.reconstruction_loss_)

        if self.weights_path != None:
            self.train_on_batch(X[:1], X[:1])

            # Load the state of the old model
            self.load_weights(self.weights_path)
            self.weights_path = None

        super().fit(
            x=X,
            y=X,
            shuffle=self.shuffle,
            epochs=self.n_iter,
            batch_size=self.batch_size_,
            validation_split=self.validation_fraction,
            callbacks=self.callbacks_,
            verbose=self.verbose,
        )

        return self

    def transform(self, X: Union[np.ndarray, tf.Tensor]):

        self.compile(optimizer=self.solver_, loss=self.reconstruction_loss_)

        if self.weights_path != None:
            self.train_on_batch(X[:1], X[:1])

            # Load the state of the old model
            self.load_weights(self.weights_path)
            self.weights_path = None

        return pipe(
            X,
            *self.encoder_hidden_tranformations,
            self.latent_layer,
            self.probability_layer
        ).mean()

    def inverse_transform(self, X: Union[np.ndarray, tf.Tensor]):
        return pipe(
            X,
            *self.decoder_hidden_tranformations,
            self.decoder_reconstruction_transformation,
            self.decoder_reconstruction_probability
        ).mean()
