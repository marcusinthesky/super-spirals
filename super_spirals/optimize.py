import tensorflow as tf

@tf.function(experimental_relax_shapes=True)
def binary_search(objective_fn, target, position_tolerance=tf.constant(1e-2), max_iterations=100,
                  lower=tf.constant(1e-20), upper=tf.constant(2000.)):
    
    lower = tf.ones(tf.shape(target)[0]) * lower
    upper = tf.ones(tf.shape(target)[0]) * upper
    
    stopping_condition = lambda x_n_1, x_n_2, f_x_n_1, target: tf.math.reduce_all(tf.abs(f_x_n_1 - target) > position_tolerance)
    
    def binary_update(x_n_1, x_n_2, f_x_n_1, target):
        test = f_x_n_1 < target
        middle = (x_n_1 + x_n_2) / 2.
        
        x_n_1 = tf.where(test, middle, x_n_1)
        x_n_2 = tf.where(test, x_n_2, middle)
                        
        return x_n_1, x_n_2, objective_fn((x_n_1 + x_n_2) / 2.), target
    
    
    x_n_1, x_n_2, f_x_n_1, target = tf.while_loop(cond = stopping_condition,
                                                   body = binary_update,
                                                   loop_vars = (lower, 
                                                                upper, 
                                                                objective_fn((lower + upper) / 2), 
                                                                target),
                                                   maximum_iterations = tf.constant(max_iterations))
    return (x_n_1 + x_n_2) / 2.

def test__binary_search():
    tf.debugging.assert_near(binary_search(lambda x: (x-1.)**2., tf.zeros((1,))), 1., 1e-1)

@tf.function
def secant_root(objective_fn: callable = tf.square, 
                target: tf.Tensor = 0.,
                initial_position: tf.Tensor = 1., 
                position_tolerance: float = 1e-08, 
                max_iterations= 100):
   
    
    stopping_condition = lambda x_n_1, x_n_2, f_x_n_1, f_x_n_2: tf.math.reduce_any(tf.abs(f_x_n_2 - target) > tf.constant(position_tolerance))
    
    def secant_update(x_n_1, x_n_2, f_x_n_1, f_x_n_2):
    
        x_n = x_n_1 - tf.math.divide_no_nan(f_x_n_1 * (x_n_1 - x_n_2), (f_x_n_1 - f_x_n_2))

        return x_n, x_n_1, objective_fn(x_n), f_x_n_1
    
    
    f_x_n_1, f_x_n_2, x_n_1, x_n_2 = tf.while_loop(cond = stopping_condition,
                                                   body = secant_update,
                                                   loop_vars = (initial_position, 
                                                                initial_position*2., 
                                                                objective_fn(initial_position), 
                                                                objective_fn(initial_position*2.)),
                                                   maximum_iterations = tf.constant(max_iterations))
        
    return x_n_2

def test__secant_root():
    tf.debugging.assert_near(secant_root(objective_fn = tf.square, 
                                         target = 0.,), 0., 1e-3)