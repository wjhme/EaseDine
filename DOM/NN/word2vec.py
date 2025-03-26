# -*- coding: utf-8 -*-
# word2vec.py:生成word2vec模型

import os
import sys
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
import codecs

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in codecs.open(os.path.join(self.dirname, fname),"r", encoding="gbk"):
                yield line.strip().split()

# word2vec.txt数据的地址
train_path = "../rawData/"

# 生成的word2vec模型的地址
model_path = "model"
sentences = MySentences(train_path)

# 此处min_count=5代表5元模型，size=100代表词向量维度，worker=15表示15个线程
model = Word2Vec(sentences,min_count = 5,vector_size=200,workers=15)

#保存模型
model.save(os.path.join(model_path, 'Word2vec_model.pkl'))
print("Model saved successfully!")