import tensorflow as tf

## TODO: Paste your implementations in here for autograder...

class MyLSTM(tf.keras.layers.Layer):
    pass

import tensorflow as tf

class MyGRU(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        super(MyGRU, self).__init__(**kwargs)
        
    def build(self, input_shape):
        kernel_shape = tf.TensorShape((input_shape[-1], 3*self.units))

        # Create trainable weight variables for this layer.
        self.kernel = self.add_weight(
            name="kernel",                shape=kernel_shape, dtype=tf.float32,
            initializer="glorot_uniform", trainable=True)
        
        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",      shape=kernel_shape, dtype = tf.float32,
            initializer="orthogonal",     trainable=True)
        
        self.bias = self.add_weight(
            name = "bias",                shape=kernel_shape, dtype=tf.float32,
            initializer = "zeros",        trainable=True)
        
        # Make sure to call the `build` method at the end
        super(MyGRU, self).build(input_shape)
        
    def call(self, inputs, initial_state = None):
        ## Hidden state 
        if initial_state is None:
            ht = tf.zeros(shape=(inputs.shape[0], self.units), dtype=tf.float32)
        else:
            ht = tf.identity(initial_state)
        
        ## Weights and biases
        W, U, b, units = self.kernel, self.recurrent_kernel, self.bias, self.units
        W_z, W_r, W_h = (W[:, :units], W[:, units:(2*units)], W[:, (2*units):])
        U_z, U_r, U_h = (U[:, :units], U[:, units:(2*units)], U[:, (2*units):])
        b = tf.reduce_sum(b, axis=0)
        b_z, b_r, b_h = (b[:units], b[units:(units*2)], b[(units*2):])
        
        outputs = [] ## we need the whole sequence of outputs
        inputs_time_major = tf.transpose(inputs, perm = [1, 0, 2]) ## swap the batch and timestep axes

        for input_each_step in inputs_time_major:

            zt = tf.sigmoid(tf.matmul(input_each_step, W_z) + tf.matmul(ht, U_z) + b_z)
            
            rt = tf.sigmoid(tf.matmul(input_each_step, W_r) + tf.matmul(ht, U_r) + b_r)
            
            hhat = tf.tanh(tf.matmul(input_each_step, W_h) + (rt * tf.matmul(ht, U_h)) + b_h)
            
            ht = (zt * ht) + ((1 - zt) * hhat)
            outputs += [ht]
        
        outputs = tf.stack(outputs, axis=0)
        
        outputs = tf.transpose(outputs, perm = [1, 0, 2])
        
        return outputs, ht
    
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.units
        return tf.TensorShape(shape)
    
    def get_config(self):
        base_config = super(MyGRU, self).get_config()
        base_config["units"] = self.units
        return base_config