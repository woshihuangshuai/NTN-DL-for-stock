#!/usr/bin/env python
# coding=utf-8


'''
    reference:
        dense_tesnor, https://github.com/bstriner/dense_tensor;
        keras-neural-tensor-network, https://github.com/dapurv5/keras-neural-tensor-layer;
'''
import scipy.stats as stats
from keras import backend as K
from keras import regularizers
from keras.engine.topology import Layer
from keras.regularizers import ActivityRegularizer, Regularizer


class NeuralTensorLayer(Layer):
    '''
    每个张量层中包含两部分：一个张量和一个标准的前向传播神经网络
    '''
    def __init__(self, output_dim, input_dim=None, W_regularizer=None, V_regularizer=None,
                 b_regularizer=None, activity_regularizer=None, **kwargs):
        self.output_dim = output_dim  # k
        self.input_dim = input_dim  # d

        self.W_regularizer = regularizers.get(W_regularizer)
        self.V_regularizer = regularizers.get(V_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(NeuralTensorLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        mean = 0.0
        std = 1.0

        # W : k*d*d
        k = self.output_dim
        d = self.input_dim

        initial_W_values = stats.truncnorm.rvs(
            -2 * std, 2 * std, loc=mean, scale=std, size=(k, d, d))
        initial_V_values = stats.truncnorm.rvs(
            -2 * std, 2 * std, loc=mean, scale=std, size=(2 * d, k))

        self.W = K.variable(initial_W_values)  # neural tensor network's parameters
        self.V = K.variable(initial_V_values)  # feed-forward neural network's parameters
        self.b = K.zeros((self.output_dim,))  # bias parameters
        self.trainable_weights = [self.W, self.V, self.b]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.V_regularizer:
            self.V_regularizer.set_param(self.V)
            self.regularizers.append(self.V_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.built = True

    def call(self, inputs, mask=None):
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('BilinearTensorLayer must be called on a list of tensors '
                            '(at least 2). Got: ' + str(inputs))
        e1 = inputs[0]
        e2 = inputs[1]
        batch_size = K.shape(e1)[0]
        k = self.output_dim
        # print([e1,e2])

        feed_forward_product = K.dot(K.concatenate([e1, e2]), self.V)
        # print(feed_forward_product)

        bilinear_tensor_products = []
        for i in range(k):
            bilinear_tensor_products.append(K.sum(e2 * K.dot(e1, self.W[i])))

        result = K.tanh(K.reshape(K.concatenate(
            bilinear_tensor_products, axis=0), (batch_size, k)) + feed_forward_product + self.b)
        # print(result)
        return result

    def get_output_shape_for(self, input_shape):
        # print (input_shape)
        batch_size = input_shape[0][0]
        return (batch_size, self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'input_dim': self.input_dim,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'V_regularizer': self.V_regularizer.get_config() if self.V_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None}
        base_config = super(NeuralTensorLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def contrastive_max_margin(y_true, y_pred):
    '''
        contrastive max-margin loss function
        目标函数缺少正则化项
    '''
    return K.mean(K.maximum(1. - y_true + y_pred, 0.), axis=-1)
