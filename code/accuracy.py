import tensorflow as tf


class RPrecision(tf.keras.metrics.Accuracy):
    def __init__(self, name='RPrecision', **kwargs):
        super(RPrecision, self).__init__(name=name)

    def call(prediction, truth, *args, **kwargs):
        '''
        :param prediction: an ordered list of song ids
        :param truth: an ordered list of song ids
        :return: the r-precision score
        '''
        prediction = set(prediction[:len(truth)])
        truth = set(truth)
        return len(prediction.intersection(truth))/len(truth)
    
class SongAccuracy(tf.keras.losses.SparseCategoricalCrossentropy):  
    def __init__(self, name='perplexity', from_logits=False, **kwargs):
        super(SongAccuracy, self).__init__(name=name, from_logits=from_logits)
    
    def call(self, *args, **kwargs):
        losses = super().call(*args, **kwargs)
        return tf.math.exp(tf.math.reduce_mean(losses))

    