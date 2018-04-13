#!/usr/bin/env python
# -*- coding: utf-8 -*-


import csv
import datetime
import glob
import os

import numpy as np
from keras import backend as K
from keras.layers import (Conv2D, Dense, Flatten, Input, MaxPooling2D, Merge,
                          Reshape)
from keras.models import Model
from keras.utils import np_utils

from prettytable import PrettyTable


'''
    short_term_input    = U1
    middle_term_input   = [U1 ~ U7]
    long_term_input     = [U1 ~ U30]
'''

def get_event_embedding_dic():
    event_embedding_dic = {}  # ntn网络产生的中间结果
    with open('../../data/ntn_result', 'r') as ntn_result_file:
        line = ntn_result_file.readline()
        while line:
            items = line.split()
            event_embedding_dic[items[0]] = items[1:]
            line = ntn_result_file.readline()
    return event_embedding_dic


def get_historical_price_trend(historical_price_data_dir):
    stock_trend_dic = {}  # 每天的股票价格趋势
    with open(historical_price_data_dir, 'r') as historical_price_data_file:
        historical_price_data_csv_file = csv.reader(
            historical_price_data_file)
        header = historical_price_data_csv_file.next()
        line = historical_price_data_csv_file.next()
        while line:
            date_time = datetime.datetime.strptime(
                line[0], '%Y-%m-%d').strftime('%Y%m%d')
            closing_price = float(line[-3])
            # 前一天的价格
            try:
                line = historical_price_data_csv_file.next()
            except StopIteration:
                break
            the_day_before_closing_price = float(line[-3])
            trend = [
                1, 0] if closing_price > the_day_before_closing_price else [0, 1]
            stock_trend_dic[date_time] = trend
    return stock_trend_dic


def generate_data_sequence(event_embedding_dic, stock_trend_dic):
    # 开始的三十天因为历史新闻数据不足，所以从30天后开始
    EM_length = 10 # 事件向量的长度
    input_data = []  # 输入数据
    label_data = []  # 标签
    date_period = 30

    min_date = min(event_embedding_dic.keys())
    start_date = datetime.datetime.strptime(
        min_date, '%Y%m%d') + datetime.timedelta(days=30)
    start_date = start_date.strftime('%Y%m%d')
    for date_time_str in stock_trend_dic.keys():
        if date_time_str < start_date:
            continue
        thirty_days_EM = []
        current_date = datetime.datetime.strptime(date_time_str, '%Y%m%d')
        for i in range(date_period):
            date_idx = (
                current_date - datetime.timedelta(days=(i + 1))).strftime('%Y%m%d')
            if date_idx in event_embedding_dic.keys():
                thirty_days_EM.append(event_embedding_dic[date_idx])
            else:
                thirty_days_EM.append([0.0] * EM_length)
        input_data.append(thirty_days_EM)
        label_data.append(stock_trend_dic[date_time_str])
    return input_data, label_data


def get_train_and_test_data_sequence(event_embedding_dic, stock_trend_dic, data_split=0.2):
    input_data, label_data = generate_data_sequence(event_embedding_dic, stock_trend_dic)
    t = int(len(label_data) * (1-data_split))
    train_input_sequence = input_data[:t]
    train_label_sequence = label_data[:t]
    test_input_sequence = input_data[t:]
    test_label_sequence = label_data[t:]

    # train data
    train_input1 = np.array([item[0] for item in train_input_sequence])
    train_input2 = np.array([item[:7] for item in train_input_sequence])
    train_input3 = np.array(train_input_sequence)
    train_label = np.array(train_label_sequence)
    train_input_array_list = [train_input1, train_input2, train_input3]

    # test data
    test_input1 = np.array([item[0] for item in test_input_sequence])
    test_input2 = np.array([item[:7] for item in test_input_sequence])
    test_input3 = np.array(test_input_sequence)
    test_label = np.array(test_label_sequence)
    test_input_array_list = [test_input1, test_input2, test_input3]

    return train_input_array_list, train_label, test_input_array_list, test_label
    

def deepPredictionModel(input_dim=3, output_dim=2):
    '''return a deep prediction model(CNN)'''

    # short term input layer
    short_term_input = Input(shape=(input_dim,),
                             dtype='float32')      # shape=(1, k)

    # middle term input layer
    middle_term_input = Input(shape=(7, input_dim),
                              dtype='float32')      # shape=(7, k)
    middle_reshape_layer = Reshape((1, 7, input_dim))(middle_term_input)
    # middle convolutional layer
    middle_conv_layer = Conv2D(nb_filter=1, nb_row=3, nb_col=1,
                               dim_ordering='th', border_mode='valid', init='glorot_normal')(middle_reshape_layer)
    # middle max-pooling layer
    middle_pooling_layer = MaxPooling2D(pool_size=(
        5, 1), border_mode='valid', dim_ordering='th')(middle_conv_layer)
    middle_term_output = Flatten()(middle_pooling_layer)

    # long term input layer
    long_term_input = Input(shape=(30, input_dim),
                            dtype='float32')        # shape=(30, k)
    long_reshape_layer = Reshape((1, 30, input_dim))(long_term_input)
    # long term convolutional layer
    long_conv_layer = Conv2D(nb_filter=1,  nb_row=3, nb_col=1,
                             dim_ordering='th', border_mode='valid', init='glorot_normal')(long_reshape_layer)
    # long term max-pooling layer
    long_pooling_layer = MaxPooling2D(pool_size=(
        28, 1), border_mode='valid', dim_ordering='th')(long_conv_layer)
    long_term_output = Flatten()(long_pooling_layer)

    # merge layer
    merge_layer = Merge(mode='concat', concat_axis=-1)(
        [short_term_input, middle_term_output, long_term_output])

    # fully-connected layer
    hidden_layer = Dense(output_dim=10, activation='sigmoid', init='glorot_normal')(merge_layer)

    # output layer
    output_layer = Dense(output_dim=2, activation='softmax', init='glorot_normal')(
        hidden_layer)  # class: Up([1, 0]); Down([0, 1])

    model = Model(input=[short_term_input, middle_term_input,
                         long_term_input], output=output_layer)

    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    model = deepPredictionModel(input_dim=10)
    model.summary()

    event_embedding_dic = get_event_embedding_dic()

    historical_price_data_file_list = ['../../data/historical_price_data/SP500.csv']
    historical_price_data_dir = '../../data/historical_price_data/'
    sub_folder_list = ['high_rank', 'middle_rank', 'low_rank']
    for sub_folder in sub_folder_list:
        for f in glob.glob(historical_price_data_dir + sub_folder + '/*'):
            historical_price_data_file_list.append(f)

    result_table = PrettyTable(
        ['Price data', 'Accuracy 1', 'Accuracy 2', 'Accuracy 3', 'Average accuracy'])

    for historical_price_data_file in historical_price_data_file_list:
        stock_trend_dic = get_historical_price_trend(historical_price_data_file)
        train_input_array_list, train_label, test_input_array_list, test_label \
         = get_train_and_test_data_sequence(event_embedding_dic, stock_trend_dic)
        
        filename = '/'.join(historical_price_data_file.split('/')[-2:]).split('.')[0]
        accuracy = [0.0] * 3
        for i in range(3): # 对于一个数据集进行三次预测求平均准确率
            model.fit([train_input_array_list[0], train_input_array_list[1], train_input_array_list[2]],
                    train_label, batch_size=100, nb_epoch=500, verbose=1)
            result = model.predict(
                [test_input_array_list[0], test_input_array_list[1], test_input_array_list[2]])
            total = float(len(test_label))
            correct = 0.0
            for item in zip(result, test_label.tolist()):
                result, label = item
                if result[0] > result[1]:
                    result = [1, 0]
                else:
                    result = [0, 1]
                if result == label:
                    correct += 1
            accuracy[i] = correct / total
        average_accuracy = sum(accuracy) / len(accuracy)
        result_table.add_row([filename, accuracy[0], accuracy[1], accuracy[2], average_accuracy])
    # print result_table
    # 将预测结果保存到文件中
    result_file_dir = '../../data/predict_result/'
    if os.path.exists(result_file_dir) == False:
        os.makedirs(result_file_dir)
    with open(result_file_dir + 'CNN_predict_result', 'w') as result_file:
        result_file.write(result_table.get_string())
