#!/usr/bin/env python2

# ****** Deprecated ******
# ****** Deprecated ******
# ****** Deprecated ******

# from keras import backend as K
# from keras.engine.topology import Layer
# from keras.regularizers import l2
# from keras.layers import initializations
# import numpy as np

# class Tensorlayer(Layer):
#     """Tensorlayer used for NTN&DL for stock experiment"""
#     def __init__(self, output_dim, **kwargs):
#         self.output_dim = output_dim    # number of slices of the Tensorlayer
#         self.W_regularizer = l2(0.0001)
#         super(Tensorlayer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.W = self.add_weight(shape=(self.output_dim, input_shape[-1], input_shape[-1]),
#                                 initializer='uniform',
#                                 trainable=True)
#         self.b = None
#         self.built = True

#     def call(self, x, mask=None):
#         """ e_1 * Tensorlayer * e_2.T """
#         t = np.array(x)
#         print t
#         return K.dot(x, self.W)

#     def get_output_shape_for(self, input_shape):
#         """ output_shape: 1 * k """
#         return (input_shape[0], self.output_dim)


# def contrastive_max_margin(y_true, y_pred):
#     return K.mean(K.maximum(1. - y_true + y_pred, 0.), axis=-1)