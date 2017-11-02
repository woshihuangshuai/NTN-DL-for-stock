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
        self.time_period = []

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

            self.time_period.append(filename.split('.')[0])
            yield input1, input2, input3  # input1中每一条数据包含了从当天的新闻标题和新闻正文中提取到的所有时间

    def get_time_period(self):
        return self.time_period


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
    '''

    ntn_result_file_dir = '../../data/ntn_result'

    model = neuralTensorNetwork()
    model.summary()

    dataGenerator = TrainDataGenerator()
    for input1, input2, input3 in dataGenerator:
        label = model.predict_on_batch(
            [np.array(input1), np.array(input2), np.array(input3)])
        random.shuffle(input1)
        model.train_on_batch(
            [np.array(input1), np.array(input2), np.array(input3)], label)
    print model.get_weights()

    time_period = dataGenerator.get_time_period()
    time_idx = 0
    with open(ntn_result_file_dir, 'w') as ntn_result_file:
        for input1, input2, input3 in dataGenerator:
            label = model.predict_on_batch(
                [np.array(input1), np.array(input2), np.array(input3)])
            
            result = np.mean(label[1])
            result = (result - result.min()) / (result.max() - result.min()) # 归一化
            ntn_result_file.write(time_period[time_idx])
            for i in result.tolist():
                ntn_result_file.write(' ')    
                ntn_result_file.write(i)
            time_idx += 1
