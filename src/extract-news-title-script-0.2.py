#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
extract news' title 
date format:
filename: 
    news_date
content:
    news_title >> each line
'''

import os
import glob
from tqdm import tqdm

end_punctuation = '.?!'

cur_dir = '../data'
reuters_folder = cur_dir + '/reuters_news_title_v2'
bloomberg_folder = cur_dir + '/bloomberg_news_title_v2'

if os.path.exists(cur_dir) == False:
    os.mkdir(cur_dir)
if os.path.exists(reuters_folder) == False:
    os.mkdir(reuters_folder)
if os.path.exists(bloomberg_folder) == False:
    os.mkdir(bloomberg_folder)

# extract news in ReutersNews106521
subfolder_list = glob.glob('../data/ReutersNews106521/*')
pbar = tqdm(total=len(subfolder_list))
for subfolder in subfolder_list:
    news_date = subfolder.split('/')[-1]
    pbar.set_description('Extracting %s' % subfolder)
    pbar.update(1)
    with open(reuters_folder + '/' + news_date, 'w') as title_file:
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
                else:
                    news_title += '\n'
                title_file.write(news_title)
pbar.close()

# extract news in 20061020_20131126_bloomberg_news
subfolder_list = glob.glob('../data/20061020_20131126_bloomberg_news/*')
pbar = tqdm(total=len(subfolder_list))
for subfolder in subfolder_list:
    news_date = ''.join(subfolder.split('/')[-1].split('-'))
    pbar.set_description('Extracting %s' % subfolder)
    pbar.update(1)
    with open(bloomberg_folder + '/' + news_date, 'w') as title_file:
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
                else:
                    news_title += '\n'
                title_file.write(news_title)
pbar.close()
