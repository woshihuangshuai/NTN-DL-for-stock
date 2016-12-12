#!/usr/bin/env python

'''
extract news' title 
date format:
filename: news_date + file_number
content: news_title
'''

import os
import glob
from tqdm import tqdm

end_punctuation = '.?!'

cur_dir = '../data'
reuters_news_title_folder = cur_dir + '/Reuters_news_title'
bloomberg_news_title_folder = cur_dir + '/Bloomberg_news_title'

if os.path.exists(cur_dir) == False:
    os.mkdir(cur_dir)
if os.path.exists(reuters_news_title_folder) == False:
    os.mkdir(reuters_news_title_folder)
if os.path.exists(bloomberg_news_title_folder) == False:
    os.mkdir(bloomberg_news_title_folder)

# extract news in ReutersNews106521
file_num = 1
date_folder_list = glob.glob('../data/ReutersNews106521/*')
pbar = tqdm(total=len(date_folder_list))
for date_folder in date_folder_list:
    news_date = date_folder.split('/')[-1]
    pbar.set_description('Extracting reuters news...')
    for news_file_dir in glob.glob(date_folder + '/*'):
        news_title_file_path = reuters_news_title_folder + '/' + news_date + '-%s' % file_num
        news_title_file = open(news_title_file_path, 'w')
        news_file = open(news_file_dir, 'r')
        str  =  news_file.readline()
        if str == '-- \n':
            str = news_file.readline()
        news_title = str.split('-- ')[-1]
        news_title = news_title.strip('\n')
        if  len(news_title) == 0:
            continue
        if news_title[-1] not in end_punctuation:
            news_title += '.'
        news_title_file.write(news_title)
        news_file.close()
        news_title_file.close()
        file_num += 1
    pbar.update(1)
pbar.close()

# extract news in 20061020_20131126_bloomberg_news
file_num = 1
date_folder_list = glob.glob('../data/20061020_20131126_bloomberg_news/*')
pbar = tqdm(total=len(date_folder_list))
for date_folder in date_folder_list:
    news_date = ''.join(date_folder.split('/')[-1].split('-'))
    pbar.set_description('Extracting bloomberg news...')
    for news_file_dir in glob.glob(date_folder + '/*'):
        news_title_file_path = bloomberg_news_title_folder + '/' + news_date + '-%s' % file_num
        news_title_file = open(news_title_file_path, 'w')
        news_file = open(news_file_dir, 'r')
        str  =  news_file.readline()
        if str == '-- \n':
            str = news_file.readline()
        news_title = str.split('-- ')[-1]
        news_title = news_title.strip('\n')
        if  len(news_title) == 0:
            continue
        if news_title[-1] not in end_punctuation:
            news_title += '.'
        news_title_file.write(news_title)
        news_file.close()
        news_title_file.close()
        file_num += 1
    pbar.update(1)
pbar.close()
