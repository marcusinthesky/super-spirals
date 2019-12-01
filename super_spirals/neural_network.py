from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from toolz.curried import *
from typing import Union, Tuple, List
import tensorflow as tf
import tensorflow_probability as tfp
import warnings
from abc import ABCMeta


class VAE(tf.keras.Model, BaseEstimator, TransformerMixin):
    def __init__(self,
                 reconstruction_loss: callable = 'negative_log_likelihood',
                 elbo_weight: float = 1.0,
                 hidden_layer_sizes: Tuple[int] = (4, 2),
                 alpha: float = 0.01,
                 l1_ratio: float = 0.,
                 activation: str = 'selu',
                 n_iter:int = 20,
                 solver: Union[str, ABCMeta] = 'adam',
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
                 tol:float = 0.0001,
                 verbose: int = 1,
                 callbacks: List = [],
                ):
        super(VAE, self).__init__()
        
        # define model
        *hidden_units, latent_unit = hidden_layer_sizes
        self.encoder_hidden_tranformations = pipe(hidden_units,
                                          map(lambda d: tf.keras.layers.Dense(units=d,
                                                                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1 = l1_ratio*alpha,
                                                                                                                             l2 = (1-l1_ratio)*alpha),
                                                                              activation=activation)),
                                                 list)
        self.latent_layer = tf.keras.layers.Dense(
            tfp.layers.IndependentNormal.params_size(latent_unit),
            activation='linear',
            name="z_dist"
        )

        prior = tfp.distributions.Independent(
            tfp.distributions.Normal(loc=tf.zeros(latent_unit), scale=1),
            reinterpreted_batch_ndims=1,
        )

        self.probability_layer = tfp.layers.IndependentNormal(
            latent_unit,
            activity_regularizer=tfp.layers.KLDivergenceRegularizer(
                prior, weight=elbo_weight
            ),
        )

        self.decoder_hidden_tranformations = pipe(reversed(hidden_units),
                                          map(lambda d: tf.keras.layers.Dense(units=d,
                                                                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1 = l1_ratio*alpha,
                                                                                                                             l2 = (1-l1_ratio)*alpha),
                                                                              activation=activation)),
                                                 list)        
        
        # get optmizer
        if solver == "auto" or solver.title() == 'Adam':
            self.solver = tf.keras.optimizers.Adam(learning_rate=learning_rate_init, 
                                                    beta_1 = beta_1,
                                                    beta_2 = beta_2,
                                                    epsilon = epsilon)
        elif solver.title() == 'SGD':
            self.solver = tf.keras.optimizers.SGD(learning_rate=learning_rate_init,
                                                  momentum=momentum,
                                                  nesterov=nesterovs_momentum)
        elif type(solver) is str:
            self.solver = getattr(tf.keras.optimizers, solver)(learning_rate=learning_rate_init)
        elif type(solver) is ABCMeta:
            self.solver = solver
        else:
            warnings.warn('Not a valid optimizer, reverting to Adam')
            self.solver = tf.keras.optimizers.Adam(learning_rate=learning_rate_init, 
                                                    beta_1 = beta_1,
                                                    beta_2 = beta_2,
                                                    epsilon = epsilon)
        
        # get batch size
        if batch_size == None or batch_size == 'auto':
            self.batch_size = 32
        else:
            self.batch_size = batch_size
            
        self.validation_fraction = validation_fraction
        self.n_iter = n_iter
        self.shuffle = shuffle
        
        # Options
        if early_stopping:
            self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                               min_delta = tol,
                                                               patience = n_iter_no_change), *callbacks]
        else:
            self.callbacks = [*callbacks]
            
        self.verbose = verbose

        # get loss
        if reconstruction_loss is 'negative_log_likelihood':
            self.reconstruction_loss = lambda x, rv_x: -rv_x.log_prob(x)
        else:
            self.reconstruction_loss = lambda x, rv_x: tf.keras.losses.mse(x, rv_x.mean())
            
        self.elbo_weight = elbo_weight
        

    def call(self, inputs: Union[np.ndarray, tf.Tensor]):
        return pipe(inputs,
                    *self.encoder_hidden_tranformations,
                    self.latent_layer,
                    self.probability_layer,
                    *self.decoder_hidden_tranformations,
                    self.decoder_reconstruction_transformation,
                    self.decoder_reconstruction_probability)
                    
            
    def fit(self, X: Union[np.ndarray, tf.Tensor], y=None):
        length, reconstruction_unit = X.shape
        self.decoder_reconstruction_transformation = tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(reconstruction_unit),
                                                                           activation='linear')
        self.decoder_reconstruction_probability = tfp.layers.IndependentNormal(reconstruction_unit)
        
        self.compile(optimizer=self.solver,
                     loss=self.reconstruction_loss)
        
        super().fit(x = X, y = X,
                    shuffle = self.shuffle,
                    epochs = self.n_iter,
                    batch_size = self.batch_size,
                    validation_split = self.validation_fraction,
                    callbacks = self.callbacks,
                    verbose=self.verbose)
        
        return self
    
    def transform(self, X: Union[np.ndarray, tf.Tensor]):
        return pipe(X,
                    *self.encoder_hidden_tranformations,
                    self.latent_layer,
                    self.probability_layer).mean()

    def inverse_transform(self, X: Union[np.ndarray, tf.Tensor]):
        return pipe(X,
                    *self.decoder_hidden_tranformations,
                    self.decoder_reconstruction_transformation,
                    self.decoder_reconstruction_probability).mean()