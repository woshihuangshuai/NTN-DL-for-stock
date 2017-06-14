#!/usr/bin/env python2
# -*- coding: utf-8 -*-


from keras import backend as K
from keras.layers import (Activation, Conv2D, Dense, Flatten, Input,
                          MaxPooling2D, Merge, Permute, Reshape)
from keras.models import Model


# short_term_input = U1
# middle_term_input = [U1 ~ U7]
# long_term_input = [U1 ~ U30]


def deepPredictionModel(input_dim=3, output_dim=2):
    '''return a deep prediction model(CNN)'''

    # short term layer
    short_term_input = Input(shape=(input_dim,),
                             dtype='float32')      # shape=(1, k)

    # middle term input-conv-pooling-flatten layer
    middle_term_input = Input(shape=(7, input_dim),
                              dtype='float32')      # shape=(7, k)
    middle_reshape_layer = Reshape((1, 7, input_dim))(middle_term_input)
    middle_conv_layer = Conv2D(nb_filter=1, nb_row=3, nb_col=1, dim_ordering='th',
                               border_mode='valid')(middle_reshape_layer)
    middle_pooling_layer = MaxPooling2D(pool_size=(
        1, input_dim), border_mode='valid', dim_ordering='th')(middle_conv_layer)
    middle_flatten_layer = Flatten()(middle_pooling_layer)

    # long term input-conv-pooling-flatten layer
    long_term_input = Input(shape=(30, input_dim),
                            dtype='float32')        # shape=(30, k)
    long_reshape_layer = Reshape((1, 30, input_dim))(long_term_input)
    long_conv_layer = Conv2D(nb_filter=1,  nb_row=3, nb_col=1, dim_ordering='th',
                             border_mode='valid')(long_reshape_layer)
    long_pooling_layer = MaxPooling2D(pool_size=(
        1, input_dim), border_mode='valid', dim_ordering='th')(long_conv_layer)
    long_flatten_layer = Flatten()(long_pooling_layer)

    # merge layer
    merge_layer = Merge(mode='concat', concat_axis=-1)(
        [short_term_input, middle_flatten_layer, long_flatten_layer])

    # fully-connected layer
    hidden_layer = Dense(10, activation='relu')(merge_layer)

    # output layer
    output = Dense(2, activation=K.sigmoid)(hidden_layer)

    model = Model(input=[short_term_input, middle_term_input,
                         long_term_input], output=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def trainCNN(model):



if __name__ == '__main__':
    model = deepPredictionModel()
    model.summary()
