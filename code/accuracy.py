import tensorflow as tf
import numpy as np

class SongAccuracy(tf.keras.losses.SparseCategoricalCrossentropy):  
    def __init__(self, name='perplexity', from_logits=False, **kwargs):
        super(SongAccuracy, self).__init__(name=name, from_logits=from_logits)
    
    def call(self, *args, **kwargs):
        losses = super().call(*args, **kwargs)
        return tf.math.exp(tf.math.reduce_mean(losses))