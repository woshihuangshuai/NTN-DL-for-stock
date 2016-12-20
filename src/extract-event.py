#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import re


data_resource = ['reuters', 'bloomberg']
reverb_dict = {}
zpar_dict = {}
event_list = []


for resource in data_resource:
    print('Extracting %s...' % resource)
    with open('../data/%s_reverb_result_v1.txt' % resource, 'r') as reverb_result_file:
        c = 0
        line = reverb_result_file.readline()
        while line:
            items = line.split('\t')
            O_1 = re.sub(r'[0-9]{8}', '', items[2]).strip(' ')
            P = items[3]
            O_2 = items[4]
            datetime = items[12].split(' ')[0]
            if re.match(r'[0-9]{8}', datetime) != None:
                if datetime not in reverb_dict.keys():
                    reverb_dict[datetime] = [(O_1, P, O_2)]
                else:
                    reverb_dict[datetime].append((O_1, P, O_2))
                c += 1
            line = reverb_result_file.readline()
        # print(len(reverb_dict.keys()))
        print('total of (O_1, P, O_2): %d.' % c)

    with open('../data/%s_zpar_dep_result.txt' % resource, 'r') as zpar_result_file:
        c = 0
        item = []
        line = zpar_result_file.readline()
        while line:
            if line != '\n':
                item.append(line.strip('\n'))
                line = zpar_result_file.readline()
                continue
            else:
                datetime = ''
                sub = ''
                predicate = ''
                obj = ''

                t = item[0].split('\t')
                datetime = t[0]
                del t[0]
                if t[-1] == 'SUB':
                    sub = t[0]
                for i in range(1,len(item)):
                    t = item[i].split('\t')
                    if t[-1] == 'SUB':
                        sub = t[0]
                    elif t[-1] == 'ROOT':
                        predicate = t[0]
                    elif t[-1] == 'OBJ':
                        obj = t[0]

                if re.match(r'[0-9]{8}', datetime) != None:
                    if datetime != '' and sub != '' and predicate != '' and obj != '':
                        if datetime not in zpar_dict.keys():
                            zpar_dict[datetime] = [(sub, predicate, obj)]
                        else:
                            zpar_dict[datetime].append((sub, predicate, obj))
                        c += 1
                item = []
                line = zpar_result_file.readline()
        # print(len(reverb_dict.keys()))
        print('total of (sub, predicate, obj): %d.' % c)


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

    c = 0
    for key in reverb_dict.keys():
        reverb_list = reverb_dict[key]
        try:
            zpar_list = zpar_dict[key]
        except Exception as e:
            print 'KEY ERROR: %s.' % e
            continue
        for i in reverb_list:
            O_1 = ''.join(i[0].split(' '))
            P = ''.join(i[1].split(' '))
            O_2 = ''.join(i[2].split(' '))
            for j in zpar_list:
                if j[0] in O_1 and j[1] in P and j[2] in O_2:
                    event_list.append((key, i[0], i[1], i[2]))
                    c += 1

    f = open('../data/%s_event_list.txt' % resource, 'w')
    for i in event_list:
        f.write(str(i) + '\n')
    f.close()

    print('total of event: %d.' % c)

