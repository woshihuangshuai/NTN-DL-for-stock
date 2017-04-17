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

resources_dirs = ['../../data/ReutersNews106521/*',
                  '../../data/20061020_20131126_bloomberg_news/*']


class Mysentences(object):
    """sentences iterator"""

    def __init__(self, dirs):
        self.dirs = dirs

    def __iter__(self):
        for dir in self.dirs:
            for sub_dir in glob.glob(dir):
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
        sentences = Mysentences(resources_dirs)
        model = gensim.models.Word2Vec(sentences, size=100, min_count=1, sg=1)
        model.save('../../data/sg_model')
    return model


def event2Vec(model, event_file_list):
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
        dir_path = '../../data/event_embedding/'
        if os.path.exists(dir_path) == False:
            os.makedirs(dir_path)
        npzfile = datetime + '.npz'
        np.savez(dir_path + npzfile, event_embedding_dic[datetime])


if __name__ == '__main__':
    dir_path = '../../data/event/'
    file_list = ['bloomberg_event.txt', 'reuters_event.txt']
    event_file_list = [dir_path + file for file in file_list]
    model = getModel()
    if model != None:
        event2Vec(model, event_file_list)
