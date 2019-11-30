import tensorflow as tf

@tf.function(experimental_relax_shapes=True)
def minkowski_distance(X, Y=None, p=2):
    X_expanded = tf.expand_dims(X, -1)    
    X_tiled = tf.tile(X_expanded, [1, 1, tf.shape(X_expanded)[0]])
    
    if Y is None:
        Y_expanded = tf.transpose(X_expanded)
    else:
        Y_expanded = tf.transpose(tf.expand_dims(Y, -1))
        
    return tf.norm(X_tiled - Y_expanded, ord=p, axis=1)

def test__minkowski_distance():
    data = tf.constant([[-0.59794587,  1.084908  , -0.4533812 ],
                       [ 1.061402  , -0.41524476,  1.1109126 ],
                       [-0.1625053 ,  0.90884435,  2.1842542 ]], dtype=tf.float32)
    
    expected = tf.constant([[0.       , 2.7296352, 2.679128 ],
                           [2.7296352, 0.       , 2.0983858],
                           [2.679128 , 2.0983858, 0.       ]],
                           dtype=tf.float32)
    
    tf.debugging.assert_near(minkowski_distance(data), expected)