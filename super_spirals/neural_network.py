from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from toolz.curried import *
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.regularizers import l2
import tensorflow as tf


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.math.exp(0.5 * z_log_var) * epsilon


class VAE(BaseEstimator, TransformerMixin):
    """Transform data using vae"""

    def __init__(
        self,
        hidden_layer_sizes: tuple = (25, 2),
        activation: str = "relu",
        solver: str = "adam",
        divergence_weight: float = 1,
        alpha: float = 0.0001,
        batch_size: str = "auto",
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        power_t: float = 0.5,
        max_iter: int = 200,
        shuffle: bool = True,
        random_state=None,
        tol: float = 0.0001,
        verbose: bool = False,
        warm_start: bool = False,
        momentum: float = 0.9,
        nesterovs_momentum: bool = True,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-08,
        n_iter_no_change=10,
    ):
        """
        """
        self.alpha = alpha
        self.regularizer = l2(alpha)

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        if solver == "auto":
            self.solver = "adam"
        else:
            self.solver = solver
        self.divergence_weight = divergence_weight
        self.model = None
        self.validation_fraction = validation_fraction

    def build_encoder_(self, layers: tuple):
        # VAE model = encoder + decoder
        # build encoder model
        input_shape, *encoder_shape, latent_dim = layers

        inputs = Input(shape=(input_shape,), name="encoder_input")
        transformations = pipe(
            encoder_shape,
            map(
                lambda d: Dense(
                    units=d,
                    kernel_regularizer=self.regularizer,
                    activation=self.activation,
                )
            ),
            lambda f: compose_left(*f),
        )

        x = pipe(inputs, transformations)

        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)

        # use reparameterization trick to push the sampling out as input
        z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

        # note that "output_shape" isn't necessary with the TensorFlow backend
        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

        return inputs, encoder, z_mean, z_log_var

    def build_decoder_(self, layers: tuple):
        # build decoder model
        latent_shape, *decoder_layers, original_dim = layers

        latent_inputs = Input(shape=(latent_shape,), name="z_sampling")

        transformations = pipe(
            decoder_layers,
            map(
                lambda d: Dense(
                    units=d,
                    kernel_regularizer=self.regularizer,
                    activation=self.activation,
                )
            ),
            lambda f: compose_left(*f),
        )
        final_layer = Dense(original_dim, name="original_dim")

        outputs = pipe(latent_inputs, transformations, final_layer)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name="decoder")
        return decoder

    def build_model_(self, layers: tuple):
        """"""
        inputs, self.encoder, z_mean, z_log_var = self.build_encoder_(layers)
        self.decoder = self.build_decoder_(reversed(layers))

        outputs = pipe(inputs, self.encoder, get(2), self.decoder)
        vae = Model(inputs, outputs, name="vae_mlp")

        #         if args.mse:
        reconstruction_loss = mse(inputs, outputs)
        #         else:
        #             reconstruction_loss = binary_crossentropy(inputs,
        #                                                       outputs)

        reconstruction_loss *= layers[0]
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.math.exp(z_log_var)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = tf.reduce_mean(reconstruction_loss + self.divergence_weight * kl_loss)

        vae.add_loss(vae_loss)
        vae.compile(optimizer=self.solver)

        return vae

    def fit(self, x: np.ndarray, y: np.ndarray = None):
        """
        """
        layers = x.shape[1], *self.hidden_layer_sizes

        #         if self.model is None:
        self.model = self.build_model_(layers)

        n_samples = x.shape[0]
        if self.solver == self.solver:
            self.batch_size = n_samples
        elif self.batch_size == self.solver:
            self.batch_size = min(200, n_samples)
        else:
            if self.batch_size < 1 or self.batch_size > n_samples:
                warnings.warn(
                    "Got `batch_size` less than 1 or larger than "
                    "sample size. It is going to be clipped"
                )
            self.batch_size = np.clip(self.batch_size, 1, n_samples)

        self.model.fit(
            x,
            epochs=self.max_iter,
            batch_size=self.batch_size,
            validation_split=self.validation_fraction,
        )

        #         vae.save_weights('vae_mlp_mnist.h5')

        return self

    def transform(self, X: np.ndarray):
        """
        """
        return pipe(self.encoder.predict(X), get(0))

    def sample(self, n: int):
        """
        """
        dim = self.hidden_layer_sizes[-1]
        N = tf.random.normal(tf.zeros(dim), tf.eye(dim), size=(n,))
        return self.decoder.predict(N)

    def inverse_transform(self, Xt: np.ndarray):
        """
        """
        return self.decoder.predict(Xt)

    def predict(self, X: np.ndarray):
        """
        """
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray = None):
        """
        """
        return self.model.evaluate(X)
