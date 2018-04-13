#!/usr/bin/env python
# -*- coding=utf-8 -*-


import glob
import math
import os
import random

import numpy as np
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras.objectives import hinge
from keras.optimizers import Adadelta
from keras.regularizers import l2

from NeuralTensorLayer import NeuralTensorLayer, contrastive_max_margin


class TrainDataGenerator(object):
    '''生成训练数据, 来自新闻正文中获取的event-embedding'''

    def __init__(self):
        self.all_news_EM_dir = '../../data/event_embedding/all_news/'

    def __iter__(self):
        '''每次yield一个日期内的所有EM向量'''
        for file in glob.glob(self.all_news_EM_dir + '*'):
            filename = file.split('/')[-1]
            all_news_EM = np.load(file)
            input1 = [em[0] for em in all_news_EM]
            input2 = [em[1] for em in all_news_EM]
            input3 = [em[2] for em in all_news_EM]

            date_time = filename.split('.')[0]
            yield date_time, input1, input2, input3


class PredictDataGenerator(object):
    '''生成测试数据, 从新闻标题中获取的event-embedding'''

    def __init__(self):
        self.news_title_EM_dir = '../../data/event_embedding/news_title/'

    def __iter__(self):
        for file in glob.glob(self.news_title_EM_dir + '*'):
            filename = file.split('/')[-1]
            news_title_EM = np.load(file)

            input1 = [em[0] for em in news_title_EM]
            input2 = [em[1] for em in news_title_EM]
            input3 = [em[2] for em in news_title_EM]

            date_time = filename.split('.')[0]
            yield date_time, input1, input2, input3


def neuralTensorNetwork(input_dim=100, output_dim=10):
    '''在实现网络时，应仅设定P为输出层，待网络训练完成后，通过获取U层的输出作为结果'''
    # input layer
    input1 = Input(shape=(input_dim,), dtype='float32')
    input2 = Input(shape=(input_dim,), dtype='float32')
    input3 = Input(shape=(input_dim,), dtype='float32')

    # connect arg1 and relation
    R_1 = NeuralTensorLayer(output_dim=output_dim, input_dim=input_dim, W_regularizer=l2(0.0001),
                            V_regularizer=l2(0.0001), b_regularizer=l2(0.0001))([input1, input2])

    # connect relation and arg2
    R_2 = NeuralTensorLayer(output_dim=output_dim, input_dim=input_dim, W_regularizer=l2(0.0001),
                            V_regularizer=l2(0.0001), b_regularizer=l2(0.0001))([input2, input3])

    # U layer is used for predict.
    U = NeuralTensorLayer(output_dim=output_dim, input_dim=output_dim, W_regularizer=l2(0.0001),
                          V_regularizer=l2(0.0001), b_regularizer=l2(0.0001))([R_1, R_2])

    # p layer is used for training the network.
    p = Dense(output_dim=1)(U)

    # Use this model to train the network
    train_model = Model(input=[input1, input2, input3], output=p)
    train_model.compile(optimizer='adadelta', loss=contrastive_max_margin)

    # Use this model to get event-embedding
    layer_name = 'neuraltensorlayer_3'  # layer: U
    predict_model = Model(input=train_model.input,
                          output=train_model.get_layer(layer_name).output)

    return train_model, predict_model


def trainNTN(model):
    '''No use'''
    dataGenerator = TrainDataGenerator()
    for input1, input2, input3 in dataGenerator:
        label = model.predict_on_batch(
            [np.array(input1), np.array(input2), np.array(input3)])
        random.shuffle(input1)
        model.train_on_batch(
            [np.array(input1), np.array(input2), np.array(input3)], label)
    print model.get_weights()


if __name__ == '__main__':
    '''
        第一次训练的label如何产生：    1、使用随机初始化的网络进行一次predict
                                   2、使用随机值

        用**正确**的event-embedding产生lable
        用**错误**的event-embedding和lable训练网络

        输出结果： p层的输出：为了训练而添加的层
                 U层的输出：神经张量网络输出的结果
    '''

    ntn_input_dim = 100
    ntn_output_dim = 100

    print 'Buliding model'
    train_model, predict_model = neuralTensorNetwork(input_dim=ntn_input_dim, output_dim=ntn_output_dim)
    print 'Train model summary:'
    train_model.summary()
    print 'Predict model summary:'
    predict_model.summary()

    print 'Training model'
    word_embedding_array = np.load(
        '../../data/event_embedding/word_embedding_dictionary/news_content_word_embedding_dictionary.npy')
    # print type(word_embedding_dictionary)

    for i in range(500):  # epoch(N)=500, batch_size = X
        print 'epoch: %d' % i
        train_data_generator = TrainDataGenerator()
        for date_time, input1, input2, input3 in train_data_generator:
            label = train_model.predict_on_batch([np.array(input1), np.array(input2), np.array(input3)])
            
            array_length = len(word_embedding_array)
            idx_matrix = [np.random.randint(low=0, high=array_length, size=3).tolist() for i in range(len(input1))]
            corrupt_input1 = []
            for idx_array in idx_matrix:
                corrupt_input = np.zeros(ntn_input_dim)
                for idx in idx_array:
                    corrupt_input += word_embedding_array[idx]
                corrupt_input = corrupt_input / len(idx_array)
                corrupt_input1.append(corrupt_input)
            
            train_model.train_on_batch(
                [np.array(corrupt_input1), np.array(input2), np.array(input3)], label)
    # print train_model.get_weights()

    print 'Predicting result'
    date_list = []
    result_list = []
    predict_data_generator = PredictDataGenerator()
    for date_time, input1, input2, input3 in predict_data_generator:
        label = predict_model.predict_on_batch(
            [np.array(input1), np.array(input2), np.array(input3)])
        result = np.mean(label, axis=0) # 对当天的所有事件向量求平均
        result_list.append(result.tolist())
        date_list.append(date_time)
    result_array = np.array(result_list)
    # result_array = (result_array - result_array.min(axis=0))/(result_array.max(axis=0) - result_array.min(axis=0))  # Normalization
    result_list = result_array.tolist()

    print 'Persistence'
    ntn_result_file_dir = '../../data/ntn_result'
    with open(ntn_result_file_dir, 'w') as ntn_result_file:
        for date, result in zip(date_list, result_list):
            ntn_result_file.write(date)
            for item in result:
                ntn_result_file.write(' ')
                ntn_result_file.write(str(item))
            ntn_result_file.write('\n')
