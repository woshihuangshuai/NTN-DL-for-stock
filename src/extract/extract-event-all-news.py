#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
    从所有新闻数据中提取事件
'''

import re
import glob

reverb_dir      = '../../data/reverb/'
zpar_dir        = '../../data/zpar/depparser/'
save_dir        = '../../data/event/'

reverb_postfix  = '_reverb'
zpar_postfix    = '_zpar_dep'

total = 0
count_dic = {}  # 统计从每一对文件中提取的事件的数量

file_list = glob.glob(reverb_dir + '/*')
file_idx_list = []
for file in file_list:
    filename = file.split('/')[-1]
    file_idx = filename.split('_')[0]
    file_idx_list.append(file_idx)
print file_idx_list

for file_idx in file_idx_list:
    print('Extracting %s...' % file_idx)
    reverb_extract_set = set()  # {key, set}
    zpar_extract_set = set()

    # extract (arg1, rel, arg2) from reverb file
    with open(reverb_dir + '/' + file_idx + reverb_postfix, 'r') as reverb_file:
        c = 0
        line = reverb_file.readline()
        while line:
            items = line.split('\t')
            arg1        = re.sub(r'[^a-z]+', ' ', items[2].lower()).strip()
            relation    = re.sub(r'[^a-z]+', ' ', items[3].lower()).strip()
            arg2        = re.sub(r'[^a-z]+', ' ', items[4].lower()).strip()

            if arg1 != '' and relation != '' and arg2 != '':
                reverb_extract_set.add((arg1, relation, arg2))
                c += 1
            line = reverb_file.readline()
        # print(len(reverb_extract_set))
        print('Total of (arg1, relation, arg2) in %s: %d.' % (file_idx, c))

    # extract (sub, predicate, obj) from zpar file
    with open(zpar_dir + '/' + file_idx + zpar_postfix, 'r') as zpar_file:
        c = 0
        items = []
        line = zpar_file.readline()
        while line:
            if line != '\n':
                items.append(line.strip('\n'))
                line = zpar_file.readline()
                continue
            else:
                sub = set()
                predicate = set()
                obj = set()

                for i in range(0, len(items)):
                    t = items[i].split('\t')
                    if t[-1] == 'SUB':
                        item = re.sub(r'[^a-z]+', ' ', t[0].lower()).strip()
                        if item != '':
                            sub.add(item)

                    elif t[-1] == 'ROOT':
                        item = re.sub(r'[^a-z]+', ' ', t[0].lower()).strip()
                        if item != '':
                            predicate.add(item)

                    elif t[-1] == 'OBJ':
                        item = re.sub(r'[^a-z]+', ' ', t[0].lower()).strip()
                        if item != '':
                            obj.add(item)
                if len(sub) != 0 and len(predicate) != 0 and len(obj) != 0:
                    zpar_extract_set.add((sub, predicate, obj))
                    c += 1

                items = []
                line = zpar_file.readline()
        # print(len(reverb_extract_set))
        print('Total of (sub, predicate, obj) in %s: %d.' % (file_idx, c))

    # extract event
    event_list = set()
    for reverb_item in reverb_extract_set:
        for zpar_item in zpar_extract_set:
            is_in = 0
            for sub in zpar_item[0]:
                if sub in reverb_item[0] and sub != '':
                    is_in += 1
                    break
            for predicate in zpar_item[1]:
                if predicate in reverb_item[1] and predicate != '':
                    is_in += 1
                    break
            for obj in zpar_item[2]:
                if obj in reverb_item[2] and obj != '':
                    is_in += 1
                    break
            if is_in == 3 and reverb_item[0] != '' and reverb_item[1] != '' and reverb_item[2] != '':
                event_list.add((reverb_item[0], reverb_item[1], reverb_item[2]))

    # persisitence
    f = open(save_dir + file_idx, 'w')
    for event in event_list:
        s = event[0]
        for arg in event[1:]:
            s += ',' + arg
        f.write(s + '\n')
    f.close()
    total += len(event_list)
    print('Total of event in %s: %d.' % (file_idx, len(event_list)))
print('Total of event: %d' % total)
