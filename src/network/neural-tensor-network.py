#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from keras.layers import Activation, Dense, Input, merge
from keras.models import Model

from tensor import Tensorlayer, contrastive_max_margin

dim = 100   # number of dim of word_vector
k = 3     # number of slices of tensor

e1 = Input(shape=(1, 100), name='e1')
e2 = Input(shape=(1, 100), name='e1')
tensor_output = Tensorlayer(k)(e1)
event_embedding_output = Activation('tanh')(tensor_output)
T1 = Model(input=[e1], output=event_embedding_output)
T1.compile(optimizer='SGD', loss=contrastive_max_margin,)
T1.summary()


train_data_1 = np.random.random((1000, 1, 100))
train_data_2 = np.random.random((1000, 1, 100))
labels = np.random.random((1000, k))
T1.fit([train_data_1], labels, nb_epoch=5, batch_size=32)
