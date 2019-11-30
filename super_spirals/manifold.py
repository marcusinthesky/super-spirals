import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from toolz.curried import *
from tensorflow.keras.regularizers import l2
from super_spirals.metrics.scorer import tsne_loss
import numpy as np


class ParametricTSNE(BaseEstimator, TransformerMixin):
    """t-distributed Stochastic Neighbor Embedding.
    t-SNE [1] is a tool to visualize high-dimensional data. It converts
    similarities between data points to joint probabilities and tries
    to minimize the Kullback-Leibler divergence between the joint
    probabilities of the low-dimensional embedding and the
    high-dimensional data. t-SNE has a cost function that is not convex,
    i.e. with different initializations we can get different results.
    It is highly recommended to use another dimensionality reduction
    method (e.g. PCA for dense data or TruncatedSVD for sparse data)
    to reduce the number of dimensions to a reasonable amount (e.g. 50)
    if the number of features is very high. This will suppress some
    noise and speed up the computation of pairwise distances between
    samples. For more tips see Laurens van der Maaten's FAQ [2].
    Read more in the :ref:`User Guide <t_sne>`.
    Parameters
    ----------
    n_components : int, optional (default: 2)
        Dimension of the embedded space.
    perplexity : float, optional (default: 30)
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significanlty
        different results.
    early_exaggeration : float, optional (default: 12.0)
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them. For
        larger values, the space between natural clusters will be larger
        in the embedded space. Again, the choice of this parameter is not
        very critical. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high.
    learning_rate : float, optional (default: 200.0)
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers. If the cost function gets stuck in a bad local
        minimum increasing the learning rate may help.
    n_iter : int, optional (default: 1000)
        Maximum number of iterations for the optimization. Should be at
        least 250.
    n_iter_without_progress : int, optional (default: 300)
        Maximum number of iterations without progress before we abort the
        optimization, used after 250 initial iterations with early
        exaggeration. Note that progress is only checked every 50 iterations so
        this value is rounded to the next multiple of 50.
        .. versionadded:: 0.17
           parameter *n_iter_without_progress* to control stopping criteria.
    min_grad_norm : float, optional (default: 1e-7)
        If the gradient norm is below this threshold, the optimization will
        be stopped.
    metric : string or callable, optional
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them. The default is "euclidean" which is
        interpreted as squared euclidean distance.
    init : string or numpy array, optional (default: "random")
        Initialization of embedding. Possible options are 'random', 'pca',
        and a numpy array of shape (n_samples, n_components).
        PCA initialization cannot be used with precomputed distances and is
        usually more globally stable than random initialization.
    verbose : int, optional (default: 0)
        Verbosity level.
    random_state : int, RandomState instance or None, optional (default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.  Note that different initializations might result in
        different local minima of the cost function.
    method : string (default: 'barnes_hut')
        By default the gradient calculation algorithm uses Barnes-Hut
        approximation running in O(NlogN) time. method='exact'
        will run on the slower, but exact, algorithm in O(N^2) time. The
        exact algorithm should be used when nearest-neighbor errors need
        to be better than 3%. However, the exact method cannot scale to
        millions of examples.
        .. versionadded:: 0.17
           Approximate optimization *method* via the Barnes-Hut.
    angle : float (default: 0.5)
        Only used if method='barnes_hut'
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.
    Attributes
    ----------
    embedding_ : array-like, shape (n_samples, n_components)
        Stores the embedding vectors.
    kl_divergence_ : float
        Kullback-Leibler divergence after optimization.
    n_iter_ : int
        Number of iterations run.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.manifold import TSNE
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> X_embedded = ParametricTSNE(n_components=2).fit_transform(X)
    >>> X_embedded.shape
    (4, 2)
    References
    ----------
    [1] L.J.P. van der Maaten. Learning a Parametric Embedding by Preserving Local Structure. 
         In Proceedings of the Twelfth International Conference on Artificial Intelligence & Statistics (AI-STATS), 
         JMLR W&CP 5:384-391, 2009.
    """


    def __init__(self, perplexity=30.0, 
                 norm_order_input=1.0, 
                 norm_order_latent=2.0,
                 hidden_layer_sizes: tuple = (25, 2),
                 activation: str = "relu",
                 solver: str = "adam",
                 n_iter=1000,
                 alpha: float = 0.0001,
                 batch_size: str = "auto",
                 learning_rate: str = "constant",
                 learning_rate_init: float = 0.001,
                 power_t: float = 0.5,
                 max_iter: int = 200,
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

        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.validation_fraction = validation_fraction
        self.n_iter_without_progress = n_iter_without_progress
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state

    def build_encoder_(self, layers: tuple):
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

        x = pipe(inputs, transformations)

        z = Dense(latent_dim, name="z_mean")(x)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        # instantiate encoder model
        encoder = Model(inputs, z, name="encoder")

        return inputs, encoder, z

    def build_model_(self, layers: tuple):
        inputs, encoder, z = self.build_encoder_(layers)

        encoder.compile(optimizer=self.solver,
                        loss = lambda y_true, y_pred: tsne_loss(y_true, y_pred, p_true=4/3, p_pred=2, perpexity=30.))
        
        return encoder

    def fit(self, x: np.ndarray, y: np.ndarray = None):
        """
        """
        layers = x.shape[1], *self.hidden_layer_sizes

        #         if self.model is None:
        self.model = self.build_model_(layers)


        n_samples = x.shape[0]
        if self.batch_size == None:
            self.batch_size = 32
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

        return self

    def transform(self, X: np.ndarray):
        """
        """
        return self.model.predict(X)


