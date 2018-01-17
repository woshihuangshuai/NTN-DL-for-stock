#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
从原始新闻数据提取出每篇新闻的标题，每个数据集整合成一个文件。 

输出文件名: 
    ReutersNews106521_news_title.txt
    20061020_20131126_bloomberg_news_news_title.txt

数据内容格式:
    date \t news_title \n
'''

import glob
from tqdm import tqdm

end_punctuation = '.?!'

raw_news_dir = '../../data/raw_news/'
news_title_dir = '../../data/news_title/'
raw_news_folder = ['bloomberg', 'reuters']

# extract news title from reuters' news title
reuters_news_file = open(news_title_dir + 'reuters_news_title.txt', 'w')
subfolder_list = glob.glob(raw_news_dir + 'reuters/*')
pbar = tqdm(total=len(subfolder_list))
for subfolder in subfolder_list:
    news_date = subfolder.split('/')[-1]
    pbar.set_description('Extracting %s' % subfolder)
    pbar.update(1)
    for txt_file_dir in glob.glob(subfolder + '/*'):
        with open(txt_file_dir, 'r') as f:
            str = f.readline()
            if str == '-- \n':
                str = f.readline()
            news_title = str.split('-- ')[-1]
            news_title = news_title.strip('\n')
            if len(news_title) == 0:
                continue
            if news_title[-1] not in end_punctuation:
                news_title += '.\n'
            else:
                news_title += '\n'
            reuters_news_file.write(news_date + '\t' + news_title)
pbar.close()
reuters_news_file.close()

# extract news title from bloomberg's news title
bloomberg_news_file = open(news_title_dir + 'bloomberg_news_title.txt', 'w')
subfolder_list = glob.glob(raw_news_dir + 'bloomberg/*')
pbar = tqdm(total=len(subfolder_list))
for subfolder in subfolder_list:
    news_date = ''.join(subfolder.split('/')[-1].split('-'))
    pbar.set_description('Extracting %s' % subfolder)
    pbar.update(1)
    for txt_file_dir in glob.glob(subfolder + '/*'):
        with open(txt_file_dir, 'r') as f:
            str = f.readline()
            if str == '-- \n':
                str = f.readline()
            news_title = str.split('-- ')[-1]
            news_title = news_title.strip('\n')
            if len(news_title) == 0:
                continue
            if news_title[-1] not in end_punctuation:
                news_title += '.\n'
            else:
                news_title += '\n'
            bloomberg_news_file.write(news_date + '\t' + news_title)
pbar.close()
bloomberg_news_file.close()
