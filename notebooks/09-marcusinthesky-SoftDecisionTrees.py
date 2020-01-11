#%%
from typing import List, Tuple
from operator import add
from itertools import chain
import tensorflow as tf
import pandas as pd
import holoviews as hv
from sklearn.datasets import load_digits, load_boston
from sklearn.metrics import confusion_matrix
from super_spirals.tree import SoftDecisionTreeClassifier, SoftDecisionTreeRegressor
from toolz.curried import map, pipe, partial, compose_left, reduce


hv.extension('bokeh')

#%%
digits = load_digits()

digits_X = tf.convert_to_tensor(digits.data.astype('float32'))
digits_y = tf.convert_to_tensor(digits.target.astype('float32'))
digits_y_sparse = tf.one_hot(tf.dtypes.cast(digits_y, 'int32'), 9)

#%%
digits_tree = SoftDecisionTreeClassifier(max_depth=4, classes=9, beta=0.025)
digits_tree.fit(digits_X, digits_y_sparse)

digits_y_hat = digits_tree.predict(digits_X)

#%%
pd.DataFrame(confusion_matrix(digits_y_sparse.numpy().argmax(axis=1), 
                              digits_y_hat.argmax(axis=1)),
             index = range(1,10),
             columns = range(1,10))

#%%
lead_prob = digits_tree.leaf_probabilty(digits_X)

P = tf.concat([tf.reshape(x, (-1,1)) for x in lead_prob], 1)

pd.np.random.choice(range(1000), 5)

hv.Bars(tf.random.shuffle(P)[1,:].numpy())


#%%
prototype_labels = []
for s, p in zip(digits_tree.leaf(digits_X), digits_tree.leaf_probabilty(digits_X)):
    prototype_labels.append(s * tf.reshape(p, (-1,1)))

prototype_labels = list(map(lambda x: (tf.reduce_mean(x, 0)
                                 .numpy()
                                 .argmax(0)), 
                            prototype_labels))
prototypes = list(map(lambda x: tf.reduce_mean(x, 0), digits_tree.prototype(digits_X)))

#%%
images = []
for i, (proto, label) in enumerate(zip(prototypes, prototype_labels)):
    grid = (proto
            .numpy()
            .reshape((8,8)))
    standard_grid = (grid - grid.min())/grid.max()
    
    images.append(hv.Image(standard_grid).opts(title=f'Digit: {str(label)} ------------------------------ Leaf : {i+1}',
                                               xlabel='', ylabel='', 
                                               xaxis=None, yaxis=None))

reduce(add, images)

#%%
boston = load_boston()

boston_X = tf.convert_to_tensor(boston.data.astype('float32'))
boston_X_z_score = tf.math.divide_no_nan(boston_X - tf.reduce_mean(boston_X, 0), tf.math.reduce_std(boston_X, 0))
boston_y = tf.convert_to_tensor(boston.target.astype('float32'))

#%%
boston_tree = SoftDecisionTreeRegressor(max_depth=1, classes=1, beta=0.1, alpha=1.)
boston_tree.fit(boston_X_z_score, boston_y)

y_hat = boston_tree.predict(boston_X_z_score)

#%%
def weight_plots(index: int, leaf: tf.keras.layers.Layer) -> hv.plotting.Plot:
    return (pipe(zip(boston.feature_names, 
                    leaf.w
                        .numpy()
                        .reshape((-1,))
                        .tolist()),
                hv.Bars)
            .opts(xrotation=90, 
                  xlabel='Features', 
                  ylabel='Weights', 
                  title=f'Feature Importances at Leaf {index}'))

#%%
pipe(enumerate(boston_tree.tree[-1]), 
     map(lambda leaf: weight_plots(*leaf)),
     reduce(add))

#%%
pipe(boston_X_z_score, # the data
     boston_tree.leaf_probabilty, # compose leaf probabilities
     map(lambda x: tf.reshape(x, (-1,1))), # reshape probabilities
     list,
     partial(tf.concat, axis=1), # contatenate vector
     partial(tf.reduce_mean, axis=0), # get mean accross index
     lambda x: x.numpy(), # convert to numpy
     hv.Bars # gen polts
    ).opts(title='% sample per leaf', 
           xlabel='Leaves', 
           ylabel='%')