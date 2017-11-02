#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import csv
import datetime

import numpy as np
from keras import backend as K
from keras.layers import (Conv2D, Dense, Flatten, Input,
                          MaxPooling2D, Merge, Reshape)
from keras.models import Model


'''
    short_term_input = U1
    middle_term_input = [U1 ~ U7]
    long_term_input = [U1 ~ U30]
'''


class DataGenerator(object):
    '''
        Event-embedding数据:
        股票走势label:
    '''

    def __init__(self):
        self.ntn_result_file_dir = '../../data/ntn_result'
        self.historical_stock_data_file_dir = '../../data/SP500.csv'
        self.ntn_result = {}  # ntn网络产生的中间结果
        self.stock_trend = {}  # 每天的股票价格趋势
        self.date_period = 30  # 时间周期
        # 下面三个列表中的元素已有对应
        self.date_list = []  # 日期列表
        self.input_data = []  # 输入数据
        self.label_data = []  # 标签

    def get_data(self):
        self.parse_date_file()
        # 开始的三十天因为历史新闻数据不足，所以从30天后开始
        min_date = min(self.ntn_result.keys())
        start_date = datetime.datetime.strptime(
            min_date, '%Y%m%d') + datetime.timedelta(days=self.date_period)
        start_date = start_date.strftime('%Y%m%d')
        for date_time_str in self.stock_trend.keys():
            if date_time_str < start_date:
                continue
            self.date_list.append(date_time_str)
            thirty_days_EM = []
            current_date = datetime.datetime.strptime(date_time_str, '%Y%m%d')
            for i in range(self.date_period):
                date_idx = (
                    current_date - datetime.timedelta(days=(i + 1))).strftime('%Y%m%d')
                if date_idx in self.ntn_result.keys():
                    thirty_days_EM.append(self.ntn_result[date_idx])
                else:
                    thirty_days_EM.append([0.0, 0.0, 0.0])
            self.input_data.append(thirty_days_EM)
            self.label_data.append(self.stock_trend[date_time_str])
        return self.date_list, self.input_data, self.label_data

    def parse_date_file(self):
        with open(self.ntn_result_file_dir, 'r') as ntn_result_file:
            line = ntn_result_file.readline()
            while line:
                items = line.split()
                self.ntn_result[items[0]] = [
                    float(items[1]), float(items[2]), float(items[3])]
                line = ntn_result_file.readline()
        with open(self.historical_stock_data_file_dir, 'r') as historical_stock_data_file:
            historical_stock_data_csv_file = csv.reader(
                historical_stock_data_file)
            header = historical_stock_data_csv_file.next()
            line = historical_stock_data_csv_file.next()
            while line:
                date_time = datetime.datetime.strptime(
                    line[0], '%Y-%m-%d').strftime('%Y%m%d')
                closing_price = float(line[-3])
                # 前一天的价格
                try:
                    line = historical_stock_data_csv_file.next()
                except StopIteration:
                    break
                the_day_before_closing_price = float(line[-3])
                trend = [
                    1, 0] if closing_price > the_day_before_closing_price else [0, 1]
                self.stock_trend[date_time] = trend


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
    output = Dense(2, activation=K.sigmoid)(
        hidden_layer)  # 二维向量： 升（1， 0）；降（0， 1）

    model = Model(input=[short_term_input, middle_term_input,
                         long_term_input], output=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def trainCNN(model):
    model.fit()
    pass


if __name__ == '__main__':
    model = deepPredictionModel()
    model.summary()

    dategenerator = DataGenerator()
    date_list, input_data, label_data = dategenerator.get_data()
    
    t = int(len(input_data) * 0.9)
    input_train = input_data[:t]
    input_test = input_data[t:]
    label_train = label_data[:t]
    label_test = label_data[t:]

    # train
    input1_train = np.array([item[0] for item in input_train])
    input2_train = np.array([item[:7] for item in input_train])
    input3_train = np.array(input_train)
    label_train_array = np.array(label_train)
    model.fit([input1_train, input2_train, input3_train], label_train_array, batch_size=32, nb_epoch=10)

    # test
    input1_test = np.array([item[0] for item in input_test])
    input2_test = np.array([item[:7] for item in input_test])
    input3_test = np.array(input_test)
    label_test_array = np.array(label_test)   
    result = model.predict([input1_test, input2_test, input3_test]) 
    print result