#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import glob
import math
import os
import random

import numpy as np
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras.objectives import hinge
from keras.optimizers import SGD
from keras.regularizers import l2

from NeuralTensorLayer import NeuralTensorLayer, contrastive_max_margin


class TrainDataGenerator(object):

    def __init__(self):
        self.news_title_EM_dir = '../../data/event_embedding/news_title/'
        self.all_news_EM_dir = '../../data/event_embedding/all_news/'


    def __iter__(self):
        for file in glob.glob(self.all_news_EM_dir + '*'):
            filename = file.split('/')[-1]
            all_news_EM = np.load(file)
            input1 = [em[0] for em in all_news_EM]
            input2 = [em[1] for em in all_news_EM]
            input3 = [em[2] for em in all_news_EM]

            if os.path.exists(self.news_title_EM_dir + filename) == True:
                news_title_EM = np.load(self.news_title_EM_dir + filename)
                input1_extend = [em[0] for em in news_title_EM]
                input2_extend = [em[1] for em in news_title_EM]
                input3_extend = [em[2] for em in news_title_EM]
                input1.extend(input1_extend)
                input2.extend(input2_extend)
                input3.extend(input3_extend)

            date_time = filename.split('.')[0]
            yield date_time, input1, input2, input3  # 每次生成的input中包含了从当天的新闻标题和新闻正文中提取到的所有事件


def neuralTensorNetwork(input_dim=100, output_dim=3):
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

    model = Model(input=[input1, input2, input3], output=[p, U])

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=contrastive_max_margin,
                  optimizer=sgd, loss_weights=[1., 0.])
    return model


def trainNTN(model):
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

        train_on_batch
        predict_on_batch

        用**正确**的event-embedding产生lable
        用**错误**的event-embedding和lable训练网络

        输出结果： p层的输出：为了训练而添加的层
                 U层的输出：神经张量网络输出的结果
    '''
    model = neuralTensorNetwork()
    model.summary()
    dataGenerator = TrainDataGenerator()
    for input1, input2, input3 in dataGenerator:
        label = model.predict_on_batch(
            [np.array(input1), np.array(input2), np.array(input3)])
        random.shuffle(input1)
        model.train_on_batch(
            [np.array(input1), np.array(input2), np.array(input3)], label)
    # print model.get_weights()

    date_list = []
    result_list = []
    for date_time, input1, input2, input3 in dataGenerator:
        label = model.predict_on_batch([np.array(input1), np.array(input2), np.array(input3)])
        result = np.mean(label[1], axis=0)
        result_list.append(result.tolist())
        date_list.append(date_time)

    result_array = np.array(result_list)
    result_array = (result_array - result_array.min(axis=0))/(result_array.max(axis=0) - result_array.min(axis=0)) # 结果归一化
    result_list = result_array.tolist()

    ntn_result_file_dir = '../../data/ntn_result'
    with open(ntn_result_file_dir, 'w') as ntn_result_file:
        for date, result in zip(date_list, result_list):
            ntn_result_file.write(date)
            for item in result:
                ntn_result_file.write(' ')
                ntn_result_file.write(str(item))
            ntn_result_file.write('\n')
