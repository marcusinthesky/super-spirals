#%%
from typing import List, Tuple
from operator import add
from itertools import chain
import tensorflow as tf
import pandas as pd
import numpy as np
import holoviews as hv
from sklearn.datasets import load_digits, load_boston
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier, MLPRegressor
from super_spirals.tree import SoftDecisionTreeClassifier, SoftDecisionTreeRegressor
from toolz.curried import map, pipe, partial, compose_left, reduce


hv.extension('bokeh')

#%%
digits = load_digits()

digits_X = tf.convert_to_tensor(digits.data.astype('float32'))
digits_y = tf.convert_to_tensor(digits.target.astype('float32'))


#%%
dnn = MLPClassifier()
dnn.fit(digits_X.numpy(), digits_y.numpy())

y_pred = dnn.predict(digits_X.numpy())
digits_y_sparse = tf.one_hot(tf.dtypes.cast(tf.constant(y_pred), 'int32'), 9)



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
dnn = MLPRegressor()
dnn.fit(boston_X_z_score.numpy(), boston_y.numpy())
boston_y = tf.convert_to_tensor(dnn.predict(boston_X_z_score.numpy()))

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


# Variational Bayesian Mixture Regression
# %%

def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.0))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2*n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tf.distributions.Independent(
            tf.distributions.Normal(loc=t[..., :n],
                       scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1))
    ])

def prior(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype, trainable=False),
        tfp.layers.DistributionLambda(lambda t: tf.distributions.Independent(
            tf.distributions.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1))
    ])
event_shape = [1]
num_components = 2
params_size = tfp.layers.MixtureNormal.params_size(num_components, event_shape)

model = tf.keras.Sequential([
    tfp.layers.DenseVariational(params_size, posterior_mean_field, prior, kl_weight=1/boston_X_z_score.shape[0]),
    tfp.layers.MixtureNormal(num_components, event_shape)
])

negloglik = lambda x, rv_x: -rv_x.log_prob(x)

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=negloglik)

model.fit(boston_X_z_score, boston_y, epochs=200)



# Bayesian Mixture Regression
#%%
dims = X.shape[1]
components = 2
dtype = 'float32'
X = boston_X_z_score
y = boston_y

rv_mix_probs = tfp.distributions.Dirichlet(
    concentration=np.ones(components, dtype) / 10.,
    name='rv_mix_probs')

rv_loc = tfp.distributions.Independent(
    tfp.distributions.Normal(
        loc=tf.random.normal(shape=(dims, components), dtype=dtype),
        scale=tf.ones([dims, components], dtype)),
    reinterpreted_batch_ndims=1,
    name='rv_loc')

rv_precision = tfp.distributions.Gamma(concentration=[1.0]*components, rate=[2.0] * components)

def joint_log_prob(mix_probs, beta, sigma):
    """BGMM with priors: loc=Normal, precision=Inverse-Wishart, mix=Dirichlet.

    Args:
    observations: `[n, d]`-shaped `Tensor` representing Bayesian Gaussian
      Mixture model draws. Each sample is a length-`d` vector.
    mix_probs: `[K]`-shaped `Tensor` representing random draw from
      `SoftmaxInverse(Dirichlet)` prior.
    loc: `[K, d]`-shaped `Tensor` representing the location parameter of the
      `K` components.
    chol_precision: `[K, d, d]`-shaped `Tensor` representing `K` lower
      triangular `cholesky(Precision)` matrices, each being sampled from
      a Wishart distribution.

    Returns:
    log_prob: `Tensor` representing joint log-density over all inputs.
    """

    rv_observations = tfp.distributions.MixtureSameFamily(
      mixture_distribution=tfp.distributions.Categorical(probs=mix_probs),
      components_distribution=tfp.distributions.Normal(
          loc=X @ beta,
          scale=sigma))

    log_prob_parts = [
      rv_observations.log_prob(y), # Sum over samples.
      rv_mix_probs.log_prob(mix_probs)[..., tf.newaxis],
      rv_loc.log_prob(beta),                   # Sum over components.
      rv_precision.log_prob(sigma),  # Sum over components.
    ]

    sum_log_prob = tf.reduce_sum(tf.concat(log_prob_parts, axis=-1), axis=-1)
    # Note: for easy debugging, uncomment the following:
    # sum_log_prob = tf.Print(sum_log_prob, log_prob_parts)
    return sum_log_prob

unnormalized_posterior_log_prob = joint_log_prob#functools.partial(joint_log_prob, X=X, y=y.reshape(-1,1))

initial_state = [
    tf.fill([components],
            value=np.array(1. / components, dtype),
            name='mix_probs'),
    tf.constant(tf.random.normal(shape=(dims, components), dtype=dtype),
                name='loc'),
    tf.ones((components, ), dtype=dtype, name='chol_precision'),
]

unconstraining_bijectors = [
    tfp.bijectors.SoftmaxCentered(),
    tfp.bijectors.Identity(),
    tfp.bijectors.Softplus()
    ]

[mix_probs, loc, chol_precision], kernel_results = tfp.mcmc.sample_chain(
    num_results=2000,
    num_burnin_steps=500,
    current_state=initial_state,
    kernel=tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_posterior_log_prob,
            step_size=0.1,
            num_leapfrog_steps=5),
        bijector=unconstraining_bijectors),
    trace_fn = None,
    parallel_iterations = 7)

betas = tf.reduce_mean(loc, 0)

plots = [hv.Bars(zip(data.feature_names, betas[:,l]), label=f'Feature importance at mixture {l}').opts(xrotation=45) for l in range(2)]
reduce(add, plots)
