import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from toolz.curried import pipe, map, compose_left, partial
from super_spirals.metrics.manifold import tsne_loss, stress_loss, strain_loss, lle_loss
import numpy as np
from typing import Tuple, Union, List
from pathlib import Path
from abc import ABCMeta
import warnings


class _ParametricManifold(tf.keras.Model, BaseEstimator, TransformerMixin):
    def __init__(self,
                 manifold_loss: callable = tsne_loss,
                 hidden_layer_sizes: Tuple[int] = (4, 2),
                 alpha: float = 0.01,
                 l1_ratio: float = 0.,
                 activation: str = 'selu',
                 n_iter:int = 20,
                 solver: Union[str, ABCMeta] = 'adam',
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
                 tol:float = 0.0001,
                 verbose: int = 1,
                 callbacks: List = [],
                 weights_path: Union[str, Path, None] = None
                ):
        super(_ParametricManifold, self).__init__()
        
        # define model
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.activation = self.activation
        *hidden_units, final_unit = hidden_layer_sizes
        self.hidden_tranformations = pipe(hidden_units,
                                          map(lambda d: tf.keras.layers.Dense(units=d,
                                                                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1 = alpha*l1_ratio,
                                                                                                                             l2 = alpha*(1-l1_ratio)),
                                                                              activation=activation)))
        self.final_transformation = tf.keras.layers.Dense(final_unit, activation='linear')
        
        # get optmizer
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
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
        
        # get batch size
        self.batch_size = batch_size
        if batch_size == None or batch_size == 'auto':
            self.batch_size_ = 32
        else:
            self.batch_size_ = batch_size
            
        self.validation_fraction = validation_fraction
        self.n_iter = n_iter
        self.shuffle = shuffle
        
        # Options
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.callbacks = callbacks
        if early_stopping:
            self.callbacks_ = [tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                               min_delta = tol,
                                                               patience = n_iter_no_change), *callbacks]
        else:
            self.callbacks_ = [*callbacks]
            
        self.verbose = verbose

        # get loss
        self.manifold_loss = manifold_loss

        self.weights_path = weights_path
        
        
    def call(self, inputs: Union[np.ndarray, tf.Tensor]):
        return pipe(inputs, *self.hidden_tranformations, self.final_transformation)
    
    def fit(self, X: Union[np.ndarray, tf.Tensor], y = None):
        self.compile(optimizer=self.solver_,
                     loss=self.manifold_loss)

        if self.weights_path != None:
            self.train_on_batch(X[:1], X[:1])

            # Load the state of the old model
            self.load_weights(self.weights_path)
            self.weights_path = None

        
        super().fit(x = X, y = X,
                    shuffle = self.shuffle,
                    epochs = self.n_iter,
                    batch_size = self.batch_size_,
                    validation_split = self.validation_fraction,
                    callbacks = self.callbacks_,
                    verbose=self.verbose)
        
        return self
    
    def transform(self, X: Union[np.ndarray, tf.Tensor]):
        self.compile(optimizer=self.solver_,
                     loss=self.manifold_loss)

        if self.weights_path != None:
            self.train_on_batch(X[:1], X[:1])

            # Load the state of the old model
            self.load_weights(self.weights_path)
            self.weights_path = None
            
        return self.call(X)

class ParametricTSNE(_ParametricManifold):
    def __init__(self,
                 perplexity: float = 30.,
                 norm_order_input: float = 1.,
                 norm_order_latent: float = 2.,
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
                 callbacks: List = []):
        super(ParametricTSNE, self).__init__(manifold_loss = lambda y_true, y_pred: tsne_loss(y_true, y_pred, 
                                                                          p_true=norm_order_input, 
                                                                          p_pred=norm_order_latent, 
                                                                          perpexity=perplexity),
                                            hidden_layer_sizes = hidden_layer_sizes,
                                            alpha = alpha,
                                            l1_ratio = l1_ratio,
                                            activation = activation,
                                            n_iter = n_iter,
                                            solver = solver,
                                            batch_size = batch_size,
                                            shuffle = shuffle,
                                            learning_rate_init = learning_rate_init,
                                            validation_fraction = validation_fraction,
                                            beta_1 = beta_1,
                                            beta_2 = beta_2,
                                            epsilon = epsilon,
                                            momentum = momentum,
                                            nesterovs_momentum = nesterovs_momentum,
                                            early_stopping = early_stopping,
                                            n_iter_no_change = n_iter_no_change,
                                            tol = tol,
                                            verbose = verbose,
                                            callbacks = callbacks)


class ParametricMDS(_ParametricManifold):
    def __init__(self,
                 norm_order_input: float = 1.,
                 norm_order_latent: float = 2.,
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
                 callbacks: List = [],):
        super(ParametricMDS, self).__init__(manifold_loss = lambda y_true, y_pred: stress_loss(y_true, y_pred, 
                                                                            p_true=norm_order_input, 
                                                                            p_pred=norm_order_latent),
                                            hidden_layer_sizes = hidden_layer_sizes,
                                            alpha = alpha,
                                            l1_ratio = l1_ratio,
                                            activation = activation,
                                            n_iter = n_iter,
                                            solver = solver,
                                            batch_size = batch_size,
                                            shuffle = shuffle,
                                            learning_rate_init = learning_rate_init,
                                            validation_fraction = validation_fraction,
                                            beta_1 = beta_1,
                                            beta_2 = beta_2,
                                            epsilon = epsilon,
                                            momentum = momentum,
                                            nesterovs_momentum = nesterovs_momentum,
                                            early_stopping = early_stopping,
                                            n_iter_no_change = n_iter_no_change,
                                            tol = tol,
                                            verbose = verbose,
                                            callbacks = callbacks)

class ParametricSammonMapping(_ParametricManifold):
    def __init__(self,
                 norm_order_input: float = 1.,
                 norm_order_latent: float = 2.,
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
                 callbacks: List = [],):
                 super(ParametricSammonMapping, self).__init__(manifold_loss = lambda y_true, y_pred: strain_loss(y_true, y_pred, 
                                                                                                                    p_true=norm_order_input, 
                                                                                                                    p_pred=norm_order_latent),
                                                                hidden_layer_sizes = hidden_layer_sizes,
                                                                alpha = alpha,
                                                                l1_ratio=l1_ratio,
                                                                activation = activation,
                                                                n_iter = n_iter,
                                                                solver = solver,
                                                                batch_size = batch_size,
                                                                shuffle = shuffle,
                                                                learning_rate_init = learning_rate_init,
                                                                validation_fraction = validation_fraction,
                                                                beta_1 = beta_1,
                                                                beta_2 = beta_2,
                                                                epsilon = epsilon,
                                                                momentum = momentum,
                                                                nesterovs_momentum = nesterovs_momentum,
                                                                early_stopping = early_stopping,
                                                                n_iter_no_change = n_iter_no_change,
                                                                tol = tol,
                                                                verbose = verbose,
                                                                callbacks = callbacks)


class ParametricLLE(_ParametricManifold):
    def __init__(self,
                 perplexity: float = 30.,
                 norm_order: float = 1.,
                 neighours: int = 10,
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
                 callbacks: List = [],):
        super(ParametricLLE, self).__init__(manifold_loss = lambda y_true, y_pred: lle_loss(y_true, y_pred, 
                                                                          minkowski=norm_order, 
                                                                          neighours=neighours),
                                            hidden_layer_sizes = hidden_layer_sizes,
                                            alpha = alpha,
                                            l1_ratio = l1_ratio,
                                            activation = activation,
                                            n_iter = n_iter,
                                            solver = solver,
                                            batch_size = batch_size,
                                            shuffle = shuffle,
                                            learning_rate_init = learning_rate_init,
                                            validation_fraction = validation_fraction,
                                            beta_1 = beta_1,
                                            beta_2 = beta_2,
                                            epsilon = epsilon,
                                            momentum = momentum,
                                            nesterovs_momentum = nesterovs_momentum,
                                            early_stopping = early_stopping,
                                            n_iter_no_change = n_iter_no_change,
                                            tol = tol,
                                            verbose = verbose,
                                            callbacks = callbacks)
