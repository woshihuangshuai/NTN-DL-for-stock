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

news_resources      = ['bloomberg', 'reuters']
dataset_dir         = '../../data/dataset/'

raw_news_dir        = '../../data/raw_news/'
processed_news_dir  = '../../data/processed_news/'


if os.path.exists(raw_news_dir) == False:
    os.makedirs(raw_news_dir)
if os.path.exists(processed_news_dir) == False:
    os.makedirs(processed_news_dir)

for news_resource in news_resources:
    folder_list = glob.glob(dataset_dir + '%s' %
                            news_resource + '/*')   # 获取目录下的所有子文件夹
    pbar = tqdm(total=len(folder_list))
    for folder in folder_list:
        folder_name = folder.split('/')[-1]
        datetime = ''.join(folder_name.split('-'))
        
        processed_news_save_dir = processed_news_dir + datetime + '/'
        if os.path.exists(processed_news_save_dir) == False:
            os.makedirs(processed_news_save_dir)

        raw_news_save_dir = raw_news_dir + datetime + '/'
        if os.path.exists(raw_news_save_dir) == False:
            os.makedirs(raw_news_save_dir)

        pbar.set_description('Processing %s in %s' % (datetime, news_resource))
        pbar.update(1)

        for file_dir in glob.glob(folder + '/*'):
            filename = file_dir.split('/')[-1]
  
            news_file_in_dataset = codecs.open(file_dir, 'r', encoding='UTF-8')
            lines = news_file_in_dataset.readlines()
            news_file_in_dataset.close()

            # 将数据集中的原始新闻文件移动到一个新的目录，目的是将reuters和bloomberg的新闻合并的一个目录下
            raw_news_file = codecs.open(
                raw_news_save_dir + filename, 'w', encoding='UTF-8')
            for line in lines:
                raw_news_file.write(line)
            raw_news_file.close()

            # 使用NLTK进行句子切分
            # content = ' '.join([line.strip().lower() for line in lines[7:]])
            content = ' '.join([line.strip() for line in lines[7:]])    # 不将句子中的所有字母转换成小写，测试结果 
            sentence_list = sent_tokenize(content)
            
            # 将处理的后新闻文件保存的一个新的目录下
            processed_news_file = codecs.open(
                processed_news_save_dir + filename, 'w', encoding='UTF-8')
            for sentence in sentence_list:
                processed_news_file.write(sentence + '\n')
            processed_news_file.close()

    pbar.close()
