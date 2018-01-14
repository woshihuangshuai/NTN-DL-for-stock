#!/usr/bin/env python
# -*- coding=utf-8 -*-

folder_dir = '../../data/news_title/'
news_title_file_name = 'reuters_news_title.txt'
event_file_name = 'event/reuters_event.txt'
compare_result_file_name = 'compare_result.txt'

event_dic = {}
with open(folder_dir + event_file_name, 'r') as event_file:
    line = event_file.readline()
    while line:
        items = line.split(',')
        if items[0] not in event_dic.keys():
            event_dic[items[0]] = ['\t'.join(items[1:])]
        else:
            event_dic[items[0]].append('\t'.join(items[1:]))
        line = event_file.readline()

title_dic = {}
with open(folder_dir + news_title_file_name, 'r') as news_title_file:
    line = news_title_file.readline()
    while line:
        items = line.split('\t')
        if items[0] not in title_dic.keys():
            title_dic[items[0]] = [items[1]]
        else:
            title_dic[items[0]].append(items[1])
        line = news_title_file.readline()

split_line = '********************%s********************\n'
news_split_line = '********************NEWS********************\n'
event_split_line = '********************EVENT********************\n'
with open(folder_dir + compare_result_file_name, 'w') as compare_result_file:
    for key in title_dic.keys():
        compare_result_file.write(split_line % key)
        compare_result_file.write(news_split_line)
        for news in title_dic[key]:
            compare_result_file.write(news)
        compare_result_file.write(event_split_line)
        if key in event_dic.keys():
            for event in event_dic[key]:
                compare_result_file.write(event)
