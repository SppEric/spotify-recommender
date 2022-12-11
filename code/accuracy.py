import tensorflow as tf
import numpy as np

# Perplexity Metric from Assignment 4
class SongAccuracy(tf.keras.losses.SparseCategoricalCrossentropy):  
    def __init__(self, name='Perplexity', from_logits=False, **kwargs):
        super(SongAccuracy, self).__init__(name=name, from_logits=from_logits)
    
    def call(self, *args, **kwargs):
        losses = super().call(*args, **kwargs)
        return tf.math.exp(tf.math.reduce_mean(losses))


# # RPrecision Metric from Spotify
# class RPrecision(tf.keras.metrics):
#     def __init__(self, name='RPrecision', **kwargs):
#         super(RPrecision, self).__init__(name=name)

#     def call(self, prediction, labels, *args, **kwargs):
#         '''
#         :param prediction: an ordered list of song ids
#         :param truth: an ordered list of song ids
#         :return: the r-precision score
#         '''
#         prediction = set(prediction[:len(labels)])
#         truth = set(labels)

#         return len(prediction.intersection(truth))/len(truth)