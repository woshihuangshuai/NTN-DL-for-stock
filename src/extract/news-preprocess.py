#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
    对bloomberg和reuters中的所有新闻按文件进行预处理，每篇新闻保存到单独的文件。
'''

import os
import glob
from tqdm import tqdm


news_resources = ['bloomberg', 'reuters']
news_resources_dir = '../../data/'
processed_news_dir = '../../data/processed_news/'


for news_resource in news_resources:
    save_dir = processed_news_dir + news_resource + '/'
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    folder_list = glob.glob(news_resources_dir + '%s' %
                            news_resource + '/*')   # 获取目录下的所有子文件夹
    pbar = tqdm(total=len(folder_list))

    for folder in folder_list:
        pbar.set_description('Extracting %s' % folder)
        pbar.update(1)

        for file in glob.glob(folder + '/*'):
            raw_news_file = open(file, 'r')
            processed_news_file = open(save_dir + file.split('/')[-1], 'w')

            for i in range(7):
                raw_news_file.readline()

            line = raw_news_file.readline()
            while line:
                processed_news_file.write(line.strip('\n'))
                line = raw_news_file.readline()
            processed_news_file.write('\n')

            raw_news_file.close()
            processed_news_file.close()

    pbar.close()
