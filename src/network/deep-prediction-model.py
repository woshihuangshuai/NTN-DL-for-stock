#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# input = {date: [event-embeddings]}

from keras import backend as K
from keras.layers import Activation, Dense, Input
from keras.layers import Conv1D, MaxPooling1D, Permute
from keras.layers import Merge
from keras.models import Model

input_dim = 3


def deepPredictionModel(input_dim=3, output_dim=2):
    short_term_input = Input(shape=(1, input_dim),
                             dtype='float32')      # shape=(1, k)
    middle_term_input = Input(shape=(7, input_dim),
                              dtype='float32')      # shape=(7, k)
    long_term_input = Input(shape=(30, input_dim),
                            dtype='float32')     # shape=(30, k)

    # input shape = (k, 7)  output shape = (k, 5)
    # middle_term_input_transpose = Permute((2, 1))(middle_term_input)
    middle_conv = Conv1D(nb_filter=1, filter_length=3,
                         border_mode='valid')(middle_term_input)
    middle_conv_pooling = MaxPooling1D(pool_length=5, border_mode='valid')(middle_conv)

    # input shape = (k, 30)  output shape = (k, 28)
    long_term_input_transpose = Permute((2, 1))(long_term_input)
    long_conv = Conv1D(nb_filter=1, filter_length=3, border_mode='valid')(
        long_term_input_transpose)
    long_conv_pooling = MaxPooling1D(pool_length=28, border_mode='valid')

    conv_pooling_layer_output = Merge(mode='concat', concat_axis=-1)([short_term_input, middle_conv_pooling, long_conv_pooling])

    hidden_layer = Dense(30, activation='relu')(conv_pooling_layer_output)
    output = Dense(2, activation=K.sigmoid)(hidden_layer)

    model = Model(input=[short_term_input, middle_term_input,
                         long_term_input], output=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.summary()

    return model


if __name__ == '__main__':
    deepPredictionModel()
