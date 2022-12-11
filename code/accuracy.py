import tensorflow as tf

# Perplexity Metric from Assignment 4
class SongAccuracy(tf.keras.losses.SparseCategoricalCrossentropy):  
    def __init__(self, name='Perplexity', from_logits=False, **kwargs):
        super(SongAccuracy, self).__init__(name=name, from_logits=from_logits)
    
    def call(self, *args, **kwargs):
        losses = super().call(*args, **kwargs)
        return tf.math.exp(tf.math.reduce_mean(losses))

    
