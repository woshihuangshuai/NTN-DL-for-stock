#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import glob
import logging
import os
import re
import string

import gensim

# from tqdm import tqdm
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

resources_dirs = ['../data/ReutersNews106521/*',
                  '../data/20061020_20131126_bloomberg_news/*']

# intab = string.punctuation \
#             + '～！@＃¥％……&＊（）｛｝［］｜、；：‘’“”，。／？《》＝＋－——｀' \
#             + '！‘’“”#￥%（）*+，-。、：；《》=？@【】·~——{|} '
# outtab = ' '*len(intab)
# transtab = string.maketrans(intab, outtab)


class Mysentences(object):
    """sentences iterator"""

    def __init__(self, dirs):
        self.dirs = dirs

    def __iter__(self):
        for dir in self.dirs:
            for sub_dir in glob.glob(dir):
                for txt in glob.glob(sub_dir + '/*'):
                    f = open(txt, 'r')
                    line = f.readline()
                    while line:
                        # line = re.subn(r'[\n\r\t]+', '', line)[0]        # 去除字符串中间的空格
                        # line = re.subn(r'[0-9]+', '', line)[0]        # 去除字符串中间的空格
                        # line = line.translate(transtab).strip()          #
                        # 去除字符串中的标点符号和位于字符串左边和右边的空格
                        line = re.sub(r'[^a-z]+', ' ', line.lower()).strip()
                        line = [word.strip() for word in line.split()]
                        if len(line) > 0:
                            yield line
                        line = f.readline()
                    f.close()


if __name__ == '__main__':
    if os.path.exists('../data/sg_model'):
        model = gensim.models.Word2Vec.load('../data/sg_model')
    else:
        sentences = Mysentences(resources_dirs)
        model = gensim.models.Word2Vec(sentences, size=100, min_count=5, sg=1)
        model.save('../data/sg_model')
