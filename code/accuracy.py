import tensorflow as tf


class RPrecision(tf.keras.metrics):
    def __init__(self, name='RPrecision', **kwargs):
        super(RPrecision, self).__init__(name=name)

    def call(self, prediction, truth, *args, **kwargs):
        '''
        :param prediction: an ordered list of song ids
        :param truth: an ordered list of song ids
        :return: the r-precision score
        '''
        prediction = set(prediction[:len(truth)])
        truth = set(truth)
        return len(prediction.intersection(truth))/len(truth)

# class NDCG(tf.keras.metrics):
#     def __init__(self, name='NDCG', **kwargs):
#         super(NDCG, self).__init__(name=name)
#
#     def call(self, prediction, truth, *args, **kwargs):
#         '''
#         :param prediction: an ordered list of song ids
#         :param truth: an ordered list of song ids
#         :return: the normalized discounted cumulative gain
#         '''
#         dcg = tf.einsum
#         return False
