#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
    对bloomberg和reuters中的所有新闻按文件进行预处理，每篇新闻保存到单独的文件。
'''

import os
import glob
from tqdm import tqdm


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

        for file in glob.glob(folder + '/*'):
            filename=file.split('/')[-1]
            raw_news_file=open(file, 'r')

            processed_news_file=open(save_dir + filename, 'w')

            content=''
            for i in range(7):
                raw_news_file.readline()

            line=raw_news_file.readline()
            while line:
                content += line.strip('\n')
                line=raw_news_file.readline()
            news_lines=content.split('. ')
            # print news_lines
            for line in news_lines:
                processed_news_file.write(line + '. \n')

            raw_news_file.close()
            processed_news_file.close()
    pbar.close()
