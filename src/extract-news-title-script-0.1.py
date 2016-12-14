#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
extract news' title 
date format:
filename: 
    ReutersNews106521_news_title.txt
    20061020_20131126_bloomberg_news_news_title.txt
content format:
    date \t news_title \n
'''

import glob
from tqdm import tqdm

end_punctuation = '.?!'

# extract news in ReutersNews106521
reuters_news_file = open('../data/reuters_news_title.txt', 'w')
subfolder_list = glob.glob('../data/ReutersNews106521/*')
pbar = tqdm(total=len(subfolder_list))
for subfolder in subfolder_list:
    news_date = subfolder.split('/')[-1]
    pbar.set_description('Extracting %s' % subfolder)
    pbar.update(1)
    for txt_file_dir in glob.glob(subfolder + '/*'):
        with open(txt_file_dir, 'r') as f:
            str  =  f.readline()
            if str == '-- \n':
                str = f.readline()
            news_title = str.split('-- ')[-1]
            news_title = news_title.strip('\n')
            if  len(news_title) == 0:
                continue
            if news_title[-1] not in end_punctuation:
                news_title += '.\n'
            reuters_news_file.write(news_date + '\t' + news_title)
pbar.close()
reuters_news_file.close()

# extract news in 20061020_20131126_bloomberg_news
bloomberg_news_file = open('../data/bloomberg_news_title.txt', 'w')
subfolder_list = glob.glob('../data/20061020_20131126_bloomberg_news/*')
pbar = tqdm(total=len(subfolder_list))
for subfolder in subfolder_list:
    news_date = ''.join(subfolder.split('/')[-1].split('-'))
    pbar.set_description('Extracting %s' % subfolder)
    pbar.update(1)
    for txt_file_dir in glob.glob(subfolder + '/*'):
        with open(txt_file_dir, 'r') as f:
            str  =  f.readline()
            if str == '-- \n':
                str = f.readline()
            news_title = str.split('-- ')[-1]
            news_title = news_title.strip('\n')
            if  len(news_title) == 0:
                continue
            if news_title[-1] not in end_punctuation:
                news_title += '.\n'
            bloomberg_news_file.write(news_date + '\t' + news_title)
pbar.close()
bloomberg_news_file.close()