#!/usr/bin/env python

'''
extract news' title 
date format:
date >> filename
news title >> each line
'''

import os
import glob
from tqdm import tqdm

cur_dir = '../dataset'
reuters_folder = cur_dir + '/Reuters_news_title'
bloomberg_folder = cur_dir + '/Bloomberg_news_title'

if os.path.exists(cur_dir) == False:
    os.mkdir(cur_dir)
if os.path.exists(reuters_folder) == False:
    os.mkdir(reuters_folder)
if os.path.exists(bloomberg_folder) == False:
    os.mkdir(bloomberg_folder)

# extract news in ReutersNews106521
subfolder_list = glob.glob('../dataset/ReutersNews106521/*')
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
                title_file.write(news_title)
pbar.close()

# extract news in 20061020_20131126_bloomberg_news
subfolder_list = glob.glob('../dataset/20061020_20131126_bloomberg_news/*')
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
                title_file.write(news_title)
pbar.close()
