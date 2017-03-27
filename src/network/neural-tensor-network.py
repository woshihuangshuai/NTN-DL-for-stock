#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import math

import numpy as np
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras.objectives import hinge
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.datasets import load_digits

from NeuralTensorLayer import NeuralTensorLayer, contrastive_max_margin


def get_data():
    digits = load_digits()
    L = int(math.floor(digits.data.shape[0] * 0.15))
    X_train = digits.data[:L]
    y_train = digits.target[:L]
    X_test = digits.data[L + 1:]
    y_test = digits.target[L + 1:]
    return X_train, y_train, X_test, y_test


def neuralTensorNetwork(input_dim=100, output_dim=3):
    input1 = Input(shape=(input_dim,), dtype='float32')
    input2 = Input(shape=(input_dim,), dtype='float32')
    input3 = Input(shape=(input_dim,), dtype='float32')

    R_1 = NeuralTensorLayer(output_dim=output_dim, input_dim=input_dim, W_regularizer=l2(0.0001),
                            V_regularizer=l2(0.0001), b_regularizer=l2(0.0001))([input1, input2])
    R_2 = NeuralTensorLayer(output_dim=output_dim, input_dim=input_dim, W_regularizer=l2(0.0001),
                            V_regularizer=l2(0.0001), b_regularizer=l2(0.0001))([input2, input3])
    U = NeuralTensorLayer(output_dim=output_dim, input_dim=output_dim, W_regularizer=l2(0.0001),
                          V_regularizer=l2(0.0001), b_regularizer=l2(0.0001))([R_1, R_2])

    p = Dense(output_dim=1)(U)  # this layer is used for training the network.

    model = Model(input=[input1, input2, input3], output=[p, U])
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=contrastive_max_margin,
                  optimizer=sgd, loss_weights=[1., 0.])
    return model


def main():
    input1 = Input(shape=(64,), dtype='float32')
    input2 = Input(shape=(64,), dtype='float32')
    input3 = Input(shape=(64,), dtype='float32')

    # NTN = NeuralTensorLayer(output_dim=32, input_dim=64, W_regularizer=l2(0.0001),
    #                         V_regularizer=l2(0.0001), b_regularizer=l2(0.0001))([input1, input2])
    # p = Dense(output_dim=1)(NTN)
    # model = Model(input=[input1, input2], output=[p])
    # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss=contrastive_max_margin, optimizer=sgd)

    model = neuralTensorNetwork(input_dim=64, output_dim=32)

    X_train, Y_train, X_test, Y_test = get_data()
    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    Y_test = Y_test.astype(np.float32)

    model.fit([X_train, X_train, X_train], [
              Y_train, Y_train], nb_epoch=50, batch_size=5)
    score = model.evaluate([X_test, X_test, X_test], [
                           Y_test, Y_test], batch_size=1)
    print score
    # print K.get_value(model.layers[2].W)


if __name__ == '__main__':
    main()
