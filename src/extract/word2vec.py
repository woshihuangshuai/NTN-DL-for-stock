#!/usr/bin/env python
# -*- coding: utf-8 -*-


import glob
import logging
import os
import re
import string
import codecs

import gensim
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import trange

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Mysentences(object):
    """sentences iterator"""

    def __init__(self):
        self.news_dir = '../../data/merged_raw_news/'

    def __iter__(self):
        for news_dir in glob.glob(self.news_dir + '/*'):
            with codecs.open(news_dir, 'r', encoding='UTF-8') as f:
                line = f.readline()
                while line:
                    words = [re.sub(r'[^a-z]+', ' ', word).strip()
                             for word in word_tokenize(line)]  # 去除 除了小写字符a到z之外的所有字符，包括特殊符号和数字
                    words = [word for word in words if len(
                        word) > 1]  # 去除长度小于2的单词
                    if len(words) > 0:
                        yield words
                    line = f.readline()


def get_word2vec_model():
    '''从文件中导入或训练生成Word2Vec模型'''
    model = None
    if os.path.exists('../../data/sg_model/sg_model'):
        model = gensim.models.Word2Vec.load('../../data/sg_model/sg_model')
    else:
        sentences = Mysentences()
        model = gensim.models.Word2Vec(sentences, size=100, min_count=5, sg=1)
        model.save('../../data/sg_model/sg_model')
    return model


def get_event_embedding_from_news_title_event():
    print 'Transforming news title event to word_embedding.'
    source_dir = '../../data/news_title/event/*'
    save_dir = '../../data/event_embedding/news_title/'

    event_file_list = [file for file in glob.glob(source_dir)]
    model = get_word2vec_model()

    if model != None:
        event2VecNewsTitle(model, event_file_list, save_dir)
    else:
        print 'ERROR: Can\'t get event2vec model!'


def event2VecNewsTitle(model, event_file_list, save_dir):
    '''
        file format:
            filename:   <datetime>.npz
            content:    [ 
                            [   
                                [...],
                                [...],
                                [...]   # a word-embedding 
                            ],

                            [   
                                [...],
                                [...],
                                [...]   # a word-embedding 
                            ],

                            ...

                            # word-embeddings of one day's events 
                        ]
    '''

    word_embedding_dic = {}
    event_embedding_dic = {}

    for file in event_file_list:
        print 'Transforming %s event into word-embedding.' % file
        with codecs.open(file, 'r', encoding='UTF-8') as event_file:
            line = event_file.readline()
            while line:
                event_embedding = []
                args = line.strip().split(',')
                if len(args) != 4:
                    line = event_file.readline()
                    continue
                datetime = args[0]

                for arg in args[1:]:
                    # 去除 除了小写字符a到z之外的所有字符，包括特殊符号和数字
                    words = [re.sub(r'[^a-z]+', ' ', word).strip()
                             for word in word_tokenize(arg)]
                    # 去除长度小于2的单词
                    words = [word for word in words if len(
                        word) > 1]

                    length = len(words)
                    sum = np.zeros(100)
                    for word in words:
                        try:
                            word_embedding = model.wv[word]
                            sum += word_embedding
                            if word not in word_embedding_dic.keys():
                                word_embedding_dic[word] = word_embedding
                        except:
                            length -= 1
                            continue
                    if length > 0:
                        mean = sum / length
                    else:
                        mean = np.random.randn(100)
                        for i in range(len(mean)):
                            mean[i] = mean[i] - int(mean[i])
                    event_embedding.append(mean)

                if datetime not in event_embedding_dic.keys():
                    event_embedding_dic[datetime] = [event_embedding]
                else:
                    event_embedding_dic[datetime].append(event_embedding)

                line = event_file.readline()

    # # Normalization
    # all_word_embeddings = []
    # for key in event_embedding_dic.keys():
    #     all_word_embeddings.extend(
    #         [word_embedding for word_embedding in event_embedding_dic[key]])

    # all_word_embeddings_array = np.array(all_word_embeddings)
    # min_word_embedding = all_word_embeddings_array.min(axis=0)
    # max_word_embedding = all_word_embeddings_array.max(axis=0)
    # max_minus_min = max_word_embedding - min_word_embedding

    # for key in event_embedding_dic.keys():
    #     for i in range(len(event_embedding_dic[key])):
    #         t = event_embedding_dic[key][i]
    #         t = (t - min_word_embedding) / max_minus_min
    #         event_embedding_dic[key][i] = t

    # persistence
    print 'Persisting...'
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    for datetime in event_embedding_dic.keys():
        npzfile = datetime + '.npy'
        np.save(save_dir + npzfile, event_embedding_dic[datetime])

    # save word-embedding dictionary
    word_embedding_dic_save_dir = '../../data/event_embedding/word_embedding_dictionary/'
    if os.path.exists(word_embedding_dic_save_dir) == False:
        os.makedirs(word_embedding_dic_save_dir)

    word_embedding_list = []
    for key in word_embedding_dic.keys():
        word_embedding_list.append(word_embedding_dic[key])
    dic_npzfile = 'news_title_word_embedding_dictionary.npy'
    np.save(word_embedding_dic_save_dir + dic_npzfile, word_embedding_list)


def get_event_embedding_from_news_content_event():
    print 'Transforming all news event to event_embedding.'
    news_dir = '../../data/event/*'
    save_dir = '../../data/event_embedding/all_news/'

    event_file_list = [file for file in glob.glob(
        news_dir) if len(file.split('/')[-1]) == 8]
    model = get_word2vec_model()

    if model != None:
        event2VecAllNews(model, event_file_list, save_dir)
    else:
        print 'ERROR: Can\'t get event2vec model!'


def event2VecAllNews(model, event_file_list, save_dir):
    '''
        file format:
            filename:   <datetime>.npz
            content:    [ 
                            [   
                                [...],
                                [...],
                                [...]   # a event-embedding 
                            ],

                            [   
                                [...],
                                [...],
                                [...]   # a event-embedding 
                            ],

                            ...

                            # event-embeddings of one day 
                        ]
    '''

    word_embedding_dic = {}
    event_embedding_dic = {}

    for file_idx in trange(len(event_file_list), desc='Main progress'):
        file = event_file_list[file_idx]
        filename = file.split('/')[-1]
        event_embedding_list = []

        with codecs.open(file, 'r', encoding='UTF-8') as event_file:
            lines = event_file.readlines()
            for line_idx in trange(len(lines), desc='%s' % filename):
                line = lines[line_idx]
                event_embedding = []

                args = line.strip().split(',')
                if len(args) != 3:
                    continue

                for arg in args:
                    # 去除 除了小写字符a到z之外的所有字符，包括特殊符号和数字
                    words = [re.sub(r'[^a-z]+', ' ', word).strip()
                             for word in word_tokenize(arg)]
                    # 去除长度小于2的单词
                    words = [word for word in words if len(
                        word) > 1]

                    length = len(words)
                    sum = np.zeros(100)
                    for word in words:
                        try:
                            word_embedding = model.wv[word]
                            sum += word_embedding
                            if word not in word_embedding_dic.keys():
                                word_embedding_dic[word] = word_embedding
                        except:
                            length -= 1
                            continue
                    if length != 0:
                        mean = sum / length
                    else:
                        mean = np.random.randn(100)
                        for i in range(len(mean)):
                            mean[i] = mean[i] - int(mean[i])
                    event_embedding.append(mean)

                event_embedding_list.append(event_embedding)
                line = event_file.readline()

        event_embedding_dic[filename] = event_embedding_list

    # # Normalization
    # all_word_embeddings = []
    # for key in event_embedding_dic.keys():
    #     all_word_embeddings.extend(
    #         [word_embedding for word_embedding in event_embedding_dic[key]])

    # all_word_embeddings_array = np.array(all_word_embeddings)
    # min_word_embedding = all_word_embeddings_array.min(axis=0)
    # max_word_embedding = all_word_embeddings_array.max(axis=0)
    # max_minus_min = max_word_embedding - min_word_embedding

    # for key in event_embedding_dic.keys():
    #     for i in range(len(event_embedding_dic[key])):
    #         t = event_embedding_dic[key][i]
    #         t = (t - min_word_embedding) / max_minus_min
    #         event_embedding_dic[key][i] = t

    # persistence
    print 'persisting...'
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    for datetime in event_embedding_dic.keys():
        npzfile = datetime + '.npy'
        np.save(save_dir + npzfile, event_embedding_dic[datetime])

    # save word-embedding dictionary
    word_embedding_dic_save_dir = '../../data/event_embedding/word_embedding_dictionary/'
    if os.path.exists(word_embedding_dic_save_dir) == False:
        os.makedirs(word_embedding_dic_save_dir)

    word_embedding_list = []
    for key in word_embedding_dic.keys():
        word_embedding_list.append(word_embedding_dic[key])
    dic_npzfile = 'news_content_word_embedding_dictionary.npy'
    np.save(word_embedding_dic_save_dir + dic_npzfile, word_embedding_list)


if __name__ == '__main__':
    get_event_embedding_from_news_title_event()
    get_event_embedding_from_news_content_event()
