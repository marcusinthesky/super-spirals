#%%
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from toolz.curried import pipe, map, compose_left, partial
import numpy as np
from typing import Tuple, Union, List
from pathlib import Path
from abc import ABCMeta
import warnings
import tensorflow_probability as tfp


@tf.function
def explainer_mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor, sample_weights: tf.Tensor=None):
    length = tf.shape(y_true)[0]
    square_error = tf.square(y_pred - (y_true * tf.ones((length, length), 'float32')))
    if sample_weights is not None:
        return tf.reduce_mean(square_error*sample_weights)
    return tf.reduce_mean(square_error)

@tf.function
def factorial(x: int):
    return tf.exp(tf.math.lgamma(x))

@tf.function
def choose(n: int, k: int):
    return factorial(n) / (factorial(k) * factorial(n-k))

@tf.function
def shapely_kernel(z_prime: tf.Tensor):
    M = tf.dtypes.cast(tf.shape(z_prime)[1], z_prime.dtype)
    z_abs = tf.reshape(tf.reduce_sum(z_prime, -1), (-1,))
    return tf.math.divide_no_nan((M-1), choose(M, z_abs) * z_abs * (M - z_abs))

class Bias(tf.keras.layers.Layer):
    def __init__(self, output_dim: Union[Tuple[int], int], **kwargs):
        self.output_dim = output_dim
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape: int):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1, self.output_dim),
                                      initializer='normal',
                                      trainable=True)
        super(Bias, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x: tf.Tensor):
        return x + self.kernel

    def compute_output_shape(self, input_shape: int):
        return (input_shape[0], self.output_dim)

@tf.function
def explainer_train_step(index: tf.Tensor, X: tf.Tensor, y: tf.Tensor, sample_weights: tf.Tensor, explainer: tf.keras.Model, optimizer: tf.keras.optimizers):
    with tf.GradientTape() as tape:        
        loss = explainer_mse_loss(explainer(index, X), y, sample_weights)
    gradients = tape.gradient(loss, explainer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, explainer.trainable_variables))

class _TabularExplainer(tf.keras.Model, BaseEstimator):
    def __init__(self, 
                 kernel_fn: callable = tf.ones_like,
                 alpha: float = 1e-3,
                 l1_ratio: float = 1.,
                 n_iter:int = 1000,
                 solver: Union[str, ABCMeta] = 'adam',
                 batch_size: int = 100,
                 shuffle: bool = True,
                 learning_rate_init: float = 0.0001,
                 validation_fraction: float = 0.1,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-08,
                 momentum: float = 0.9,
                 nesterovs_momentum: bool = True):
        super(_TabularExplainer, self).__init__()
        self.kernel_fn = kernel_fn
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.batch_size = batch_size
        self.n_iter = n_iter

        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        # get optmizer
        if solver == "auto" or solver.title() == 'Adam':
            self.solver_ = tf.keras.optimizers.Adam(learning_rate=learning_rate_init, 
                                                    beta_1 = beta_1,
                                                    beta_2 = beta_2,
                                                    epsilon = epsilon)
        elif solver.title() == 'SGD':
            self.solver_ = tf.keras.optimizers.SGD(learning_rate=learning_rate_init,
                                                  momentum=momentum,
                                                  nesterov=nesterovs_momentum)
        elif type(solver) is str:
            self.solver_ = getattr(tf.keras.optimizers, solver)(learning_rate=learning_rate_init)
        elif type(solver) is ABCMeta:
            self.solver_ = solver
        else:
            warnings.warn('Not a valid optimizer, reverting to Adam')
            self.solver_ = tf.keras.optimizers.Adam(learning_rate=learning_rate_init, 
                                                    beta_1 = beta_1,
                                                    beta_2 = beta_2,
                                                    epsilon = epsilon)
        
        
        self.tranpose = tf.keras.layers.Lambda(lambda x: tf.transpose(x))
        self.dot = tf.keras.layers.Dot((0,0))
        self.bias = Bias(1)
        
    def call(self, index: tf.Tensor, X: tf.Tensor):
        weights = self.feature_importances(index)
        transpose_weights = self.tranpose(weights)

        return self.bias(X @ transpose_weights)

    def train_loop(self, batches: tf.data.Dataset):
        for epoch in range(self.n_iter):
            for sample in batches:
                index_sample, X_sample, y_sample = sample
                sample_weights = self.kernel_fn(X_sample)
                explainer_train_step(index_sample, X_sample, y_sample, sample_weights, self, self.solver_)

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        self.X_shape: Tuple = tuple(X.shape)
        self.y_shape: Tuple = (y.shape[0], 1)

        self.feature_importances = tf.keras.layers.Embedding(self.X_shape[0]+1, 
                                                             self.X_shape[1],
                                                             embeddings_initializer = 'normal',
                                                             embeddings_regularizer = tf.keras.regularizers.l1_l2(l1=self.alpha*self.l1_ratio,
                                                                                                             l2=self.alpha*(1-self.l1_ratio)))
        
        index = tf.reshape(tf.range(X.shape[0]), (-1,))

        batches = (tf.data.Dataset.from_tensor_slices((index, 
        X.astype('float32'), 
        y.astype('float32')))
        .batch(self.batch_size))

        self.train_loop(batches)
            
        return self.feature_importances(index).numpy()


class LimeTabularExplainer(_TabularExplainer):
    def __init__(self,
                 kernel_width: float = 1.,
                 alpha: float = 1e-3,
                 l1_ratio: float = 1.,
                 n_iter:int = 1000,
                 solver: Union[str, ABCMeta] = 'adam',
                 batch_size: int = 100,
                 shuffle: bool = True,
                 learning_rate_init: float = 0.0001,
                 validation_fraction: float = 0.1,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-08,
                 momentum: float = 0.9,
                 nesterovs_momentum: bool = True):

        self.kernel_width = kernel_width

        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(1., kernel_width)
        kernel_fn = lambda x: kernel.matrix(x, x)

        super(LimeTabularExplainer, self).__init__(kernel_fn = kernel_fn,
        alpha = alpha,
        l1_ratio = l1_ratio, 
        n_iter = n_iter,
        batch_size = batch_size, 
        shuffle = shuffle, 
        learning_rate_init = learning_rate_init,
        validation_fraction = validation_fraction,
        beta_1 = beta_1,
        beta_2 = beta_2,
        epsilon = epsilon,
        momentum = momentum,
        nesterovs_momentum = nesterovs_momentum)

class SHAPTabularExplainer(_TabularExplainer):
    def __init__(self,
    model,
    alpha: float = 1e-3,
    l1_ratio: float = 1.,
    n_iter:int = 1000,
    solver: Union[str, ABCMeta] = 'adam',
    batch_size: int = 100,
    shuffle: bool = True,
    learning_rate_init: float = 0.0001,
    validation_fraction: float = 0.1,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-08,
    momentum: float = 0.9,
    nesterovs_momentum: bool = True):

        self.model = model

        super(SHAPTabularExplainer, self).__init__(kernel_fn = shapely_kernel,
        alpha = alpha,
        l1_ratio = l1_ratio, 
        n_iter = n_iter,
        batch_size = batch_size, 
        shuffle = shuffle, 
        learning_rate_init = learning_rate_init,
        validation_fraction = validation_fraction,
        beta_1 = beta_1,
        beta_2 = beta_2,
        epsilon = epsilon,
        momentum = momentum,
        nesterovs_momentum = nesterovs_momentum)

    def train_loop(self, batches: tf.data.Dataset):
        for epoch in range(self.n_iter):
            for sample in batches:
                index_sample, X_sample, y_sample = sample

                z_prime = tf.dtypes.cast(tf.random.uniform(tf.shape(X_sample))>0.5, 'float32')
                z_prime_minus = tf.abs(z_prime-1)

                z_mean = tf.reduce_mean(X_sample)
                z = X_sample * z_prime + z_mean * z_prime_minus
                y_pred = tf.dtypes.cast(self.model.predict(z), 'float32')

                sample_weights = self.kernel_fn(z_prime)

                explainer_train_step(index_sample, z, y_pred, sample_weights, self, self.solver_)
