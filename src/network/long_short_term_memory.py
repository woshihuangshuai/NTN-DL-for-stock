#!/usr/bin/env python
# -*- coding=utf-8 -*-


'''
Input shape: (None, 30, 100)
Output shape: (None, 1)
'''
import numpy as np
from keras.layers.core import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential

maxlen = 30  # 如何输入长度少于30个，再前面添加0向量以凑齐30个
batch_size = 32
nb_epoch = 15

print 'Loading data...'
nb_sample = 2000
length_event_embedding = 100
test_split = 0.2

X_train = np.random.rand(
    int(nb_sample * (1 - test_split)), 30, length_event_embedding)
Y_train = np.random.randint(0, 2, int(nb_sample * (1 - test_split)))
X_test = np.random.rand(
    int(nb_sample * test_split), 30, length_event_embedding)
Y_test = np.random.randint(0, 2, int(nb_sample * test_split))

print 'Pad sequences...'
# TODO

print 'Build model...'
model = Sequential()
model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, input_shape=(30, 100)))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print model.summary()

print 'Train...'
model.fit(X_train, Y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, validation_data=(X_test, Y_test))

print 'Test...'
score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

result = model.predict(X_test)
print result
