import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from toolz.curried import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from super_spirals.metrics.manifold import tsne_loss, stress_loss, strain_loss, lle_loss
import numpy as np


class _ParametricManifold(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 manifold_loss: callable = tsne_loss,
                 hidden_layer_sizes: tuple = (25, 2),
                 activation: str = "relu",
                 solver: str = "adam",
                 n_iter=20,
                 alpha: float = 0.0001,
                 batch_size: str = "auto",
                 learning_rate: str = "constant",
                 learning_rate_init: float = 0.001,
                 power_t: float = 0.5,
                 max_iter: int = 20,
                 shuffle: bool = True,
                 random_state=None,
                 n_iter_without_progress=300,
                 init="random", verbose=0,
                 warm_start: bool = False,
                 momentum: float = 0.9,
                 nesterovs_momentum: bool = True,
                 early_stopping: bool = False,
                 validation_fraction: float = 0.1,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-08,
                 n_iter_no_change=10):

        self.manifold_loss = manifold_loss
        self.batch_size = batch_size
        self.alpha = alpha
        self.regularizer = l2(alpha)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        if solver == "auto" or solver.title() == 'Adam':
            self.solver = tf.keras.optimizers.Adam(learning_rate=learning_rate_init, 
                                                    beta_1 = beta_1,
                                                    beta_2 = beta_2,
                                                    epsilon = epsilon)
        else:
            self.solver = getattr(tf.keras.optimizers, solver.title())(learning_rate=learning_rate_init)

        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.validation_fraction = validation_fraction
        self.n_iter_without_progress = n_iter_without_progress
        self.init = init
        self.verbose = verbose
        self.random_state = random_state

    def _build_encoder(self, layers: tuple):
        # Build parametric model
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

        final_layer = pipe(inputs, transformations)
        z = Dense(latent_dim, activation='linear', name="z_mean")(final_layer)

        encoder = Model(inputs, z, name="encoder")

        return inputs, encoder, z

    def _build_model(self, layers: tuple):
        inputs, encoder, z = self._build_encoder(layers)

        encoder.compile(optimizer=self.solver,
                        loss = self.manifold_loss)
        
        return encoder

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        """
        layers = X.shape[1], *self.hidden_layer_sizes

        self.model = self._build_model(layers)


        n_samples = X.shape[0]
        if self.batch_size == None or self.batch_size == 'auto':
            self.batch_size = 32
        else:
            if self.batch_size < 1 or self.batch_size > n_samples:
                warnings.warn(
                    "Got `batch_size` less than 1 or larger than "
                    "sample size. It is going to be clipped"
                )
            self.batch_size = np.clip(self.batch_size, 1, n_samples)

        self.model.fit(
            X, X,
            epochs=self.max_iter,
            batch_size=self.batch_size,
            validation_split=self.validation_fraction,
        )

        return self

    def transform(self, X: np.ndarray):
        """
        """
        return self.model.predict(X)


class ParametricTSNE(_ParametricManifold):
    def __init__(self,
                 perplexity: float = 30.,
                 norm_order_input: float = 1.,
                 norm_order_latent: float = 2.,
                 *args, **kwargs):
        super().__init__(manifold_loss = lambda y_true, y_pred: tsne_loss(y_true, y_pred, 
                                                                          p_true=norm_order_input, 
                                                                          p_pred=norm_order_latent, 
                                                                          perpexity=perplexity),
                         *args, **kwargs)


class ParametricMDS(_ParametricManifold):
    def __init__(self,
                 perplexity: float = 30.,
                 norm_order_input: float = 1.,
                 norm_order_latent: float = 2.,
                 *args, **kwargs):
        super().__init__(manifold_loss = lambda y_true, y_pred: stress_loss(y_true, y_pred, 
                                                                            p_true=norm_order_input, 
                                                                            p_pred=norm_order_latent),
                         *args, **kwargs)

class ParametricSammonMapping(_ParametricManifold):
    def __init__(self,
                 perplexity: float = 30.,
                 norm_order_input: float = 1.,
                 norm_order_latent: float = 2.,
                 *args, **kwargs):
        super().__init__(manifold_loss = lambda y_true, y_pred: strain_loss(y_true, y_pred, 
                                                                            p_true=norm_order_input, 
                                                                            p_pred=norm_order_latent),
                         *args, **kwargs)

class ParametricLLE(_ParametricManifold):
    def __init__(self,
                 perplexity: float = 30.,
                 norm_order_input: float = 1.,
                 norm_order_latent: float = 2.,
                 *args, **kwargs):
        super().__init__(manifold_loss = lambda y_true, y_pred: lle_loss(y_true, y_pred, 
                                                                          p_true=norm_order_input, 
                                                                          p_pred=norm_order_latent),
                         *args, **kwargs)
