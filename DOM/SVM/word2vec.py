# -*- coding: utf-8 -*-
# Created by huxiaoman 2018.1.28
# word2vec.py:生成word2vec模型

import os
import sys
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
import codecs

reload(sys)
sys.setdefaultencoding( "utf-8" )

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in codecs.open(os.path.join(self.dirname, fname),"r", encoding="utf-8",errors="ignore"):
                yield line.strip().split()

# word2vec.txt数据的地址
train_path = "rawData/"

# 生成的word2vec模型的地址
model_path = "/modelPath/"
sentences = MySentences(train_path)

# 此处min_count=5代表5元模型，size=100代表词向量维度，worker=15表示15个线程
model = Word2Vec(sentences,min_count = 5,size=100,workers=15)

#保存模型
model.save(model_path+'/Word2vec_model.pkl')
