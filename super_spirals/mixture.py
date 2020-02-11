import tensorflow as tf
import tensorflow_probability as tfp


class Gamma:
    def __init__(self, concentration, rate):
        super(Gamma, self).__init__()
        
        self.concentration = concentration
        self.rate = rate
    
    def __call__(self, shape, dtype=None):
        return tf.random.gamma(shape, 
                               alpha = self.concentration, 
                               beta = self.rate, 
                               dtype=getattr(tf.dtypes, dtype))


class VariationalBayesianGaussianMixture(tf.keras.layers.Layer):
    def __init__(self, num_outputs, convert_to_tensor_fn = tfp.distributions.Distribution.sample):
        super(VariationalBayesianGaussianMixture, self).__init__()
        self.num_outputs = num_outputs
        self.convert_to_tensor_fn = convert_to_tensor_fn
        

    def build(self, input_shape):
        _input_shape = input_shape[1]
        self._input_shape = _input_shape
        
        # Variational distribution variables for means
        locs_init = tf.keras.initializers.RandomNormal()
        self.locs = tf.Variable(initial_value=locs_init(shape=(self.num_outputs, _input_shape),
                                                        dtype='float32'),
                                trainable=True)
        scales_init = Gamma(5., 5.)
        self.scales = tf.Variable(initial_value=scales_init(shape=(self.num_outputs, _input_shape),
                                                  dtype='float32'),
                                  trainable=True)

        # Variational distribution variables for standard deviations
        alpha_init = tf.keras.initializers.RandomUniform(4., 6.)
        self.alpha = tf.Variable(initial_value=alpha_init(shape=(self.num_outputs, _input_shape), dtype='float32'),
                                 trainable=True)
        beta_init = tf.keras.initializers.RandomUniform(4., 6.)
        self.beta = tf.Variable(initial_value=beta_init(shape=(self.num_outputs, _input_shape), dtype='float32'),
                                trainable=True)

        counts_init = tf.keras.initializers.Constant(2)
        self.counts = tf.Variable(initial_value=counts_init(shape=(self.num_outputs,), dtype='float32'),
                                  trainable=True)

        # priors
        mu_mu_prior = tf.zeros((self.num_outputs, _input_shape))
        mu_sigma_prior = tf.ones((self.num_outputs, _input_shape))
        self.mu_prior = tfp.distributions.Normal(mu_mu_prior, mu_sigma_prior)

        sigma_concentration_prior = 5.*tf.ones((self.num_outputs, _input_shape))
        sigma_rate_prior = 5.*tf.ones((self.num_outputs, _input_shape))
        self.sigma_prior = tfp.distributions.Gamma(sigma_concentration_prior, sigma_rate_prior)

        theta_concentration = 2.*tf.ones((self.num_outputs,))
        self.theta_prior = tfp.distributions.Dirichlet(theta_concentration)
    

    def call(self, inputs, sampling=True):
        n_samples = tf.dtypes.cast(tf.reduce_mean(tf.reduce_sum(inputs ** 0., 0)), 'float32') # TODO: get rid of expensie hack
        
        # The variational distributions
        mu = tfp.distributions.Normal(self.locs, self.scales)
        sigma = tfp.distributions.Gamma(self.alpha, self.beta)
        theta = tfp.distributions.Dirichlet(self.counts)
        
        # Sample from the variational distributions
        if sampling:
            mu_sample = mu.sample(n_samples)
            sigma_sample = tf.pow(sigma.sample(n_samples), -0.5)
            theta_sample = theta.sample(n_samples)
        else:
            mu_sample = tf.reshape(mu.mean(), (1, self.num_outputs, self._input_shape))
            sigma_sample = tf.pow(tf.reshape(sigma.mean(), (1, self.num_outputs, self._input_shape)), -0.5)
            theta_sample = tf.reshape(theta.mean(), (1, self.num_outputs))
        
        # The mixture density
        density = tfp.distributions.MixtureSameFamily(
                        mixture_distribution=tfp.distributions.Categorical(probs=theta_sample),
                        components_distribution=tfp.distributions.MultivariateNormalDiag(loc=mu_sample,
                                                                                         scale_diag=sigma_sample))
                
        # Compute the mean log likelihood
        log_likelihoods = density.log_prob(inputs)
        
        # Compute the KL divergence sum
        mu_div    = tf.reduce_sum(tfp.distributions.kl_divergence(mu,    self.mu_prior))
        sigma_div = tf.reduce_sum(tfp.distributions.kl_divergence(sigma, self.sigma_prior))
        theta_div = tf.reduce_sum(tfp.distributions.kl_divergence(theta, self.theta_prior))
        kl_sum = sigma_div + theta_div + mu_div
                
        self.add_loss(kl_sum/n_samples - log_likelihoods)
        
        return tfp.layers.DistributionLambda(lambda x: density, 
                                             convert_to_tensor_fn=self.convert_to_tensor_fn)(inputs)
