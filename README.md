# NTN-DL-for-stock

## 三方库

    keras 1.2.2
    tqdm
    gensim
    numpy
    scipy
    NLTK
    codecs

## Data
### 新闻数据
来源：Bloomberg & Reuters （Ding的论文）<br>
时间范围： 20061020 ～ 20131126 （2595天，其中1786个交易日）

### 股票价格数据
来源：https://cn.investing.com/indices/us-spx-500-historical-data<br>
时间范围： 20061020 ～ 20131126 （2595天，其中1786个交易日）

## Description

    extract-event：

        ZPar： 分割句子，提取句子中的2元关系
        ReVerb： 提取句子中的subject、predicate、object （主、谓、宾）
        ZPar和ReVerb均存在一定的错误率， 因此将两者提取的结果结合提取出event。

    word2vec:

        依赖于Gensim，基于Skig-gram实现的word2vec方法，将文本数据中的单词映射到向量空间。

    neural-tensor-network:

        神经张量网络, 对向量特征进行更高层次的抽象和映射

    CNN:

        卷积、最大池化、分类

## Tools

    ReVerb:
        java -Xmx512m -jar reverb.jar -h (for more option).

    ZPar:
        Step 1: english.postagger
        Step 2: english.depparser

## Issues
    文本数据处理：
        对于新闻文本数据，在代码中统一使用unicode编码进行所有操作

    extract-event.py:
        zpar的提取结果中大量不完整单词，提取代码有问题，需要修复。

    word2Vec.py:
        单词映射成向量的代码未完成。

    CNN

    RNN/LSTM