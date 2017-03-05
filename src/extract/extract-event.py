#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import re

data_resource = ['bloomberg', 'reuters']
total = 0

for resource in data_resource:
    reverb_dict = {}  # {key, set}
    zpar_dict = {}

    print('Extracting %s...' % resource)
    # extract (arg1, rel, arg2)
    with open('../data/%s_reverb_result_v1.txt' % resource, 'r') as reverb_result_file:
        c = 0
        line = reverb_result_file.readline()
        while line:
            items = line.split('\t')
            arg1 = re.sub(r'[^a-z]+', ' ', items[2].lower()).strip()
            relation = re.sub(r'[^a-z]+', ' ', items[3].lower()).strip()
            arg2 = re.sub(r'[^a-z]+', ' ', items[4].lower()).strip()
            datetime = items[12].split()[0]
            if re.match(r'[0-9]{8}', datetime) != None:
                if datetime not in reverb_dict.keys():
                    reverb_dict[datetime] = [(arg1, relation, arg2)]
                elif (arg1, relation, arg2) not in reverb_dict[datetime]:
                    reverb_dict[datetime].append((arg1, relation, arg2))
                c += 1
            line = reverb_result_file.readline()
        # print(len(reverb_dict.keys()))
        print('total of (arg1, relation, arg2): %d.' % c)

    # extract (sub, predicate, obj)
    with open('../data/%s_zpar_dep_result.txt' % resource, 'r') as zpar_result_file:
        c = 0
        items = []
        line = zpar_result_file.readline()
        while line:
            if line != '\n':
                items.append(line.strip('\n'))
                line = zpar_result_file.readline()
                continue
            else:
                datetime = ''
                sub = set()
                predicate = set()
                obj = set()

                t = items[0].split('\t')
                datetime = t[0]
                del t[0]
                if t[-1] == 'SUB':
                    sub.add(re.sub(r'[^a-z]+', ' ', t[0]).strip())
                for i in range(1, len(items)):
                    t = items[i].split('\t')
                    if t[-1] == 'SUB':
                        sub.add(re.sub(r'[^a-z]+', ' ', t[0]).strip())
                    elif t[-1] == 'ROOT':
                        predicate.add(re.sub(r'[^a-z]+', ' ', t[0]).strip())
                    elif t[-1] == 'OBJ':
                        obj.add(re.sub(r'[^a-z]+', ' ', t[0]).strip())

                if datetime != '' and len(sub) != 0 and len(predicate) != 0 and len(obj) != 0:
                    if re.match(r'[0-9]{8}', datetime) != None:
                        if datetime not in zpar_dict.keys():
                            zpar_dict[datetime] = [(sub, predicate, obj)]
                        elif (sub, predicate, obj) not in zpar_dict[datetime]:
                            zpar_dict[datetime].append((sub, predicate, obj))
                        c += 1
                items = []
                line = zpar_result_file.readline()
        # print(len(reverb_dict.keys()))
        print('total of (sub, predicate, obj): %d.' % c)

    # record to file
    f = open('../data/%s_reverb_extract_result.txt' % resource, 'w')
    for key in reverb_dict.keys():
        datetime = key
        l = reverb_dict[key]
        for item in l:
            s = datetime + '\t' + str(item) + '\n'
            f.write(s)
    f.close()

    f = open('../data/%s_zpar_extract_result.txt' % resource, 'w')
    for key in zpar_dict.keys():
        datetime = key
        l = zpar_dict[key]
        for item in l:
            s = datetime + '\t' + str(item) + '\n'
            f.write(s)
    f.close()

    # extract event
    event_list = set()
    for key in reverb_dict.keys():
        reverb_list = reverb_dict[key]
        try:
            zpar_list = zpar_dict[key]
        except Exception as e:
            # print 'KEY ERROR: %s.' % e
            continue
        for i in reverb_list:
            is_in = 0
            for j in zpar_list:
                for sub in j[0]:
                    if sub in i[0] and sub != '':
                        is_in += 1
                        break
                for predicate in j[1]:
                    if predicate in i[1] and sub != '':
                        is_in += 1
                        break
                for obj in j[2]:
                    if obj in i[2] and sub != '':
                        is_in += 1
                        break
                if is_in == 3 and i[0] != '' and i[1] != '' and i[2] != '':
                    event_list.add((key, i[0], i[1], i[2]))

    f = open('../data/%s_event_list.txt' % resource, 'w')
    for i in event_list:
        f.write(str(i) + '\n')
    f.close()
    total += len(event_list)
    print('total of event in %s: %d.' % (resource, len(event_list)))
print('total of event: %d' % total)
