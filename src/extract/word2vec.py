#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import glob
import logging
import os
import re
import string

import gensim
import numpy as np

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Mysentences(object):
    """sentences iterator"""

    def __init__(self):
        self.news_dir = '../../data/raw_news/'
        self.news_folders = ['bloomberg', 'reuters']

    def __iter__(self):
        for folder in self.news_folders:
            for sub_dir in glob.glob(self.news_dir + folder + '/*'):
                for txt in glob.glob(sub_dir + '/*'):
                    f = open(txt, 'r')
                    line = f.readline()
                    while line:
                        line = re.sub(r'[^a-z]+', ' ', line.lower()).strip()
                        line = [word.strip() for word in line.split()]
                        if len(line) > 0:
                            yield line
                        line = f.readline()
                    f.close()


def getModel():
    model = None
    if os.path.exists('../../data/sg_model/sg_model'):
        model = gensim.models.Word2Vec.load('../../data/sg_model/sg_model')
    else:
        sentences = Mysentences()
        model = gensim.models.Word2Vec(sentences, size=100, min_count=1, sg=1)
        model.save('../../data/sg_model')
    return model


def event2Vec(model, event_file_list, save_dir):
    '''
        file format:
            filename:   datetime
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

    event_embedding_dic = {}
    for file in event_file_list:
        print 'Transforming %s event into event-embedding.' % file
        with open(file, 'r') as event_file:
            line = event_file.readline()
            while line:
                event_embedding = []
                t = line.strip().split(',')
                datetime = t[0]
                del t[0]
                if len(t) != 3:
                    continue
                for arg in t:
                    word_list = arg.split()
                    length = len(word_list)
                    sum = np.zeros(100)
                    for word in word_list:
                        try:
                            sum += model.wv[word]
                        except:
                            continue
                    mean = sum / length
                    event_embedding.append(mean)
                # print event_embedding

                if datetime not in event_embedding_dic.keys():
                    event_embedding_dic[datetime] = [event_embedding]
                else:
                    event_embedding_dic[datetime].append(event_embedding)

                line = event_file.readline()

    # persistence
    print 'persisting...'
    for datetime in event_embedding_dic.keys():
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
        npzfile = datetime + '.npz'
        np.savez(save_dir + npzfile, event_embedding_dic[datetime])


def event2VecAllNews():
    news_dir = '../../data/event/*'
    save_dir = '../../data/event_embedding/all_news/'

    event_file_list = [file for file in glob.glob(news_dir) if len(file) == 8]
    model = getModel()

    if model != None:
        event2Vec(model, event_file_list, save_dir)
    else:
        print 'ERROR: Can\'t get event2vec model!'


def event2VecNewsTitle():
    dir_path = '../../data/event/'
    save_dir = '../../data/event_embedding/news_title/'

    event_file_list = [file for file in glob.glob(news_dir) if len(file) > 8]
    model = getModel()

    if model != None:
        event2Vec(model, event_file_list, save_dir)
    else:
        print 'ERROR: Can\'t get event2vec model!'


if __name__ == '__main__':
    event2VecAllNews()
    event2VecNewsTitle()
