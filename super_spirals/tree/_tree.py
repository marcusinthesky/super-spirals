from typing import List, Tuple, Union
from abc import ABCMeta
from pathlib import Path
from operator import add
from itertools import chain
import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from toolz.curried import pipe, map, partial, compose_left, reduce


class Node(tf.keras.layers.Layer):
    def __init__(self, units:int=1, beta:float = 0.1, l1:float = 0.1):
        super(Node, self).__init__()
        self.units = units
        self.beta = beta
        self.l1 = l1

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        
        self.b = self.add_weight(shape=(1,self.units),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs, probability):
        
        pr = tf.nn.sigmoid(tf.matmul(inputs, self.w) + self.b)
        pr_one = tf.reshape(pr, (-1,))

        
        l1 = tf.reduce_sum(tf.abs(self.w))
        self.add_loss(self.l1 * l1)
        
        p = tf.reduce_mean(pr)
        self.add_loss(self.beta * tf.keras.losses.binary_crossentropy([0.5], p))

        
        return (pr * inputs, pr_one * probability), ( (1 - pr) * inputs, (1-pr_one) * probability)

    def get_config(self):
        return {'units': self.units}

class Leaf(tf.keras.layers.Layer):

    def __init__(self, units:int =32, l1:float = 0.1):
        super(Leaf, self).__init__()
        self.units = units
        self.l1 = l1

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        
        self.b = self.add_weight(shape=(1,self.units),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs: tf.Tensor, probability):
        
        pr = tf.matmul(inputs, self.w) + self.b
        
        l1 = tf.reduce_sum(0. * tf.abs(self.w))
        
        self.add_loss(l1)
        
        return pr, probability

    def get_config(self):
        return {'units': self.units}

    
class RegressionHead(tf.keras.layers.Layer):

    def __init__(self, units:int = 8):
        super(RegressionHead, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, inputs: tf.Tensor):
        
        weighted_inputs = [x[0] * tf.reshape(x[1], (-1,1)) for x in inputs]
        
        output = reduce(tf.add, weighted_inputs)
        
        return (output)

    def get_config(self):
        return {'units': self.units}

class ClassificationHead(tf.keras.layers.Layer):

    def __init__(self, units:int = 8):
        super(ClassificationHead, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, inputs: tf.Tensor):
        
        weighted_inputs = [tf.nn.softmax(x[0]) * tf.reshape(x[1], (-1,1)) for x in inputs]
        
        output = reduce(tf.add, weighted_inputs)
        
        return (output)

    def get_config(self):
        return {'units': self.units}

class SoftDecisionTree(tf.keras.Model, BaseEstimator, RegressorMixin):

    def __init__(self, 
    max_depth:int = 3, 
    classes: int = 1, 
    beta:float = 0.01, 
    alpha:float=0., 
    l1_ratio: float = 1., 
    head: tf.keras.layers.Layer = ClassificationHead(), 
    loss:'str' = 'categorical_crossentropy',
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
    verbose: int = 1,
    weights_path = None):
        super(SoftDecisionTree, self).__init__()
        
        self.max_depth = max_depth
        self.classes = classes
        self.beta = beta
        self.alpha = alpha
        self.l1_ratio = l1_ratio

        l1 = alpha*l1_ratio
            
        self.nodes = [[Node(1, beta=beta, l1=l1) for _ in range(2**layer)] for layer in range(self.max_depth)]
        self.leaves = [[Leaf(classes, l1=l1) for _ in range(2**(self.max_depth))]]
        
        self.tree = self.nodes + self.leaves
        
        self.head = ClassificationHead() if head is None else head
        self.loss = loss
        self.verbose = verbose

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
        
        self.batch_size = batch_size
        # get batch size
        if batch_size == None or batch_size == 'auto':
            self.batch_size_ = 32
        else:
            self.batch_size_ = batch_size
            
        self.validation_fraction = validation_fraction
        self.n_iter = n_iter
        self.shuffle = shuffle

        self.weights_path = weights_path
        
    def prototype(self, inputs: tf.Tensor):
        input_to_layers = [ [[( inputs, tf.ones((tf.shape(inputs)[0],)) )]] ] + self.tree
        proto_output = reduce(lambda x, f: self.forward(x,f), input_to_layers[:-1])
        
        return [x[0] for x in list(chain(*proto_output))]
    
    def leaf_probabilty(self, inputs: tf.Tensor):
        input_to_layers = [ [[( inputs, tf.ones((tf.shape(inputs)[0],)) )]] ] + self.tree
        proto_output = reduce(lambda x, f: self.forward(x,f), input_to_layers)
        
        return [x[1] for x in proto_output]
    
    def leaf(self, inputs: tf.Tensor):
        
        input_to_layers = [ [[( inputs, tf.ones((tf.shape(inputs)[0],)) )]] ] + self.tree
        proto_output = reduce(lambda x, f: self.forward(x,f), input_to_layers)
        
        leaf_preductions = [x[0] for x in proto_output]
    
        return list(map(tf.nn.softmax, leaf_preductions))
        
    def forward(self, inputs: List[Tuple[tf.Tensor]], layer: List[tf.keras.Model]):
        inputs = list(chain(*inputs))
        joined = zip(inputs, layer)
        return [f(x[0], x[1]) for x, f in joined]
    
    def call(self, inputs: tf.Tensor):
        input_to_layers = [ [[( inputs, tf.ones((tf.shape(inputs)[0],)) )]] ] + self.tree
        
        leaf_output = reduce(lambda x, f: self.forward(x,f), input_to_layers)
        
        return self.head(leaf_output)

    def fit(self, X: Union[np.ndarray, tf.Tensor], y=None):
        length, reconstruction_unit = X.shape

        self.compile(optimizer=self.solver_,
                     loss=self.loss)

        if self.weights_path != None:
            self.train_on_batch(X[:1], X[:1])

            # Load the state of the old model
            self.load_weights(self.weights_path)
            self.weights_path = None
        
        super().fit(x = X, y = y,
                    shuffle = self.shuffle,
                    epochs = self.n_iter,
                    batch_size = self.batch_size_,
                    validation_split = self.validation_fraction,
                    verbose=self.verbose)
        
        return self
    
class SoftDecisionTreeRegressor(SoftDecisionTree):
    def __init__(self,  
    max_depth:int = 3, 
    classes: int = 1, 
    beta:float = 0.01, 
    alpha:float=0., 
    l1_ratio: float = 1.,
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
    verbose: int = 1,
    weights_path: Union[str, Path, None] = None):
        super(SoftDecisionTreeRegressor, self).__init__(max_depth = max_depth, 
        classes = classes, 
        beta = beta, 
        alpha = alpha, 
        l1_ratio = l1_ratio, 
        head=RegressionHead(), 
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
        verbose = verbose,
        weights_path = weights_path)
        
class SoftDecisionTreeClassifier(SoftDecisionTree):
    def __init__(self,  
    max_depth:int = 3, 
    classes: int = 1, 
    beta:float = 0.01, 
    alpha:float=0., 
    l1_ratio: float = 1.,
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
    verbose: int = 1,
    weights_path: Union[str, Path, None] = None):
        super(SoftDecisionTreeClassifier, self).__init__(max_depth = max_depth, 
        classes = classes, 
        beta = beta, 
        alpha = alpha, 
        l1_ratio = l1_ratio, 
        head=ClassificationHead(), 
        loss='categorical_crossentropy',
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
        verbose = verbose,
        weights_path = weights_path)