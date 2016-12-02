#!/usr/bin/env python

'''
extract news' title 
date format:
date  \t newstime \n
'''

import glob
from tqdm import tqdm


# extract news in ReutersNews106521
reuters_news_file = open('../dataset/ReutersNews106521_news_title', 'w')
subfolder_list = glob.glob('../dataset/ReutersNews106521/*')
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
            reuters_news_file.write(news_date + '\t' + news_title)
pbar.close()
reuters_news_file.close()

# extract news in 20061020_20131126_bloomberg_news
bloomberg_news_file = open('../dataset/20061020_20131126_bloomberg_news_news_title', 'w')
subfolder_list = glob.glob('../dataset/20061020_20131126_bloomberg_news/*')
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
            bloomberg_news_file.write(news_date + '\t' + news_title)
pbar.close()
bloomberg_news_file.close()