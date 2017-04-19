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


for news_resource in news_resources:
    save_dir = processed_news_dir + news_resource + '/'
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    folder_list = glob.glob(raw_news_dir + '%s' %
                            news_resource + '/*')   # 获取目录下的所有子文件夹
    pbar = tqdm(total=len(folder_list))

    file_idx = 0
    folder_idx = 1
    if os.path.exists(save_dir + '%d/' % folder_idx) == False:
        os.makedirs(save_dir + '%d/' % folder_idx)

    for folder in folder_list:
        pbar.set_description('Extracting %s' % folder.split('/')[-1])
        pbar.update(1)

        for file in glob.glob(folder + '/*'):
            raw_news_file = open(file, 'r')

            file_idx += 1
            if file_idx == 20000:
                file_idx = 0
                folder_idx += 1
                if os.path.exists(save_dir + '%d/' % folder_idx) == False:
                    os.makedirs(save_dir + '%d/' % folder_idx)

            processed_news_file = open(
                save_dir + '%d/' % folder_idx + file.split('/')[-1], 'w')

            content = ''
            for i in range(7):
                raw_news_file.readline()
            line = raw_news_file.readline()
            while line:
                content += line.strip('\n')
                line = raw_news_file.readline()
            news_lines = content.split('. ')
            # print news_lines
            for line in news_lines:
                processed_news_file.write(line + '. \n')

            raw_news_file.close()
            processed_news_file.close()

    pbar.close()