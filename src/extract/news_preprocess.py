#!/usr/bin/env python
# -*- coding=utf-8 -*-


'''
    对bloomberg和reuters中的所有新闻按文件进行预处理，并优化目录结构。

    预处理操作包括：
        1. 去除每个新闻文件中的前7行无效内容
        2. 对于每篇新闻的内容，依照‘.’进行句子切分，一个句子为一行（切分结果并不准确）

    将reuters和bloomberg两个数据集中同一日期的所有新闻文件合并到统一目录下
    
    目录结构：
    processed_news
    |
    ├── 20061020
    |   ├── news1
    |   ├── news2
    |   └── ...
    |
    ├── 20061021
    |   ├── news1
    |   ├── news2
    |   └── ...
    |
    └── ...
    
'''

import glob
import os

from nltk.tokenize import sent_tokenize
from tqdm import tqdm

import codecs

news_resources = ['bloomberg', 'reuters']
raw_news_dir = '../../data/raw_news/'
processed_news_dir = '../../data/processed_news/'

if os.path.exists(processed_news_dir) == False:
    os.makedirs(processed_news_dir)

for news_resource in news_resources:
    folder_list = glob.glob(raw_news_dir + '%s' %
                            news_resource + '/*')   # 获取目录下的所有子文件夹
    pbar = tqdm(total=len(folder_list))

    for folder in folder_list:
        folder_name = folder.split('/')[-1]
        datetime = ''.join(folder_name.split('-'))
        save_dir = processed_news_dir + datetime + '/'
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)

        pbar.set_description('Processing %s in %s' % (datetime, news_resource))
        pbar.update(1)

        for file_dir in glob.glob(folder + '/*'):
            filename = file_dir.split('/')[-1]
  
            raw_news_file = codecs.open(file_dir, 'r', encoding='UTF-8')
            lines = raw_news_file.readlines()
            raw_news_file.close()

            content = ' '.join([line.strip().lower() for line in lines[7:]])
            sentence_list = sent_tokenize(content)
            
            processed_news_file = codecs.open(save_dir + filename, 'w', encoding='UTF-8')
            for sentence in sentence_list:
                processed_news_file.write(sentence + '\n')
            processed_news_file.close()

    pbar.close()
