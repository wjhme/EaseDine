import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from Classifiers import Classifier
from sklearn.model_selection import train_test_split

# 获取当前文件的绝对路径
current_file = Path(__file__).resolve()
# 获取项目根目录（假设项目根目录是 EaseDine 的父目录）
project_root = current_file.parent.parent.parent  # 根据实际层级调整
DOM_path = current_file.parent.parent
# 将项目根目录添加到 Python 路径
sys.path.append(str(project_root))
from DOM.utils import process_data, Embeding

# data = pd.read_csv("/mnt/disk/wjh23/EaseDineDatasets/智慧养老_label/train.txt",sep="\t")
# data = process_data(data)
# train_df, test_df = train_test_split(
#         data,
#         test_size=0.15,       # 测试集比例
#         stratify=data['dom'],
#         random_state=42      # 随机种子
#     )
# train_label = train_df['dom'].tolist()
# test_label = test_df['dom'].tolist()

# train_em = Embeding(train_df)
# # Word3Vec 测试
# DIM = 100
# train_word2vec_embeding = train_em.get_word2vec(f"{str(DOM_path)}/embeding_models/word2vec/word2vec_model_0_{DIM}.bin")

# # LDA 测试
# num_topics = 20
# lda_model_path = f"{str(DOM_path)}/embeding_models/lda/lda_model_{num_topics}.model"
# lda_dictionary_path = f"{str(DOM_path)}/embeding_models/lda/lda_dictionary_{num_topics}.gensim"
# lda_corpus_path = f"{str(DOM_path)}/embeding_models/lda/lda_corpus_{num_topics}.mm"
# train_lda_embeding = train_em.get_lda(num_topics,lda_model_path,lda_dictionary_path,lda_corpus_path)

# # 按行拼接（沿 axis=1 水平拼接）
# train_features = np.hstack([train_lda_embeding, train_word2vec_embeding])
# print("train_features形状:", train_features.shape)

# test_em = Embeding(test_df)
# test_word2vec_embeding = test_em.get_word2vec(f"{str(DOM_path)}/embeding_models/word2vec/word2vec_model_0_{DIM}.bin")
# test_lda_embeding = test_em.get_lda(num_topics,lda_model_path,lda_dictionary_path,lda_corpus_path)

# # 按行拼接（沿 axis=1 水平拼接）
# test_features = np.hstack([test_lda_embeding, test_word2vec_embeding])
# print("test_features形状:", test_features.shape)

# # 模型加载
# model = Classifier()
# model.load_models()

# # 训练集评估
# train_pred = model.predict_voting(train_features)
# acc = model.evaluate(train_pred, train_label)
# print(f"训练集准确率: {acc:.4f}")
# train_df['dom_pred'] = train_pred
# train_df.to_csv("train_df.txt",sep="\t",index=False)

# # 测试集评估
# test_pred = model.predict_voting(test_features)
# acc = model.evaluate(test_pred, test_label)
# print(f"测试集准确率: {acc:.4f}")
# test_df['dom_pred'] = test_pred
# test_df.to_csv("test_df.txt",sep="\t",index=False)


# # ============================== 预测A榜数据 ====================================================
# data = pd.read_csv("/mnt/disk/wjh23/EaseDineDatasets/results/Results_1.txt",sep="\t", header=None, names=["uuid", "text", "dom", "food"]).drop(['dom','food'],axis=1)
# A_data = process_data(data, drop_duplicates = False)
# print(A_data.shape)
#
# A_em = Embeding(A_data)
# # Word3Vec
# DIM = 100
# A_word2vec_embeding = A_em.get_word2vec(f"{str(DOM_path)}/embeding_models/word2vec/word2vec_model_0_{DIM}.bin")
#
# # LDA 测试
# num_topics = 20
# lda_model_path = f"{str(DOM_path)}/embeding_models/lda/lda_model_{num_topics}.model"
# lda_dictionary_path = f"{str(DOM_path)}/embeding_models/lda/lda_dictionary_{num_topics}.gensim"
# lda_corpus_path = f"{str(DOM_path)}/embeding_models/lda/lda_corpus_{num_topics}.mm"
# A_lda_embeding = A_em.get_lda(num_topics,lda_model_path,lda_dictionary_path,lda_corpus_path)
#
# # 按行拼接（沿 axis=1 水平拼接）
# A_features = np.hstack([A_lda_embeding, A_word2vec_embeding])
# print("A_features:", A_features.shape)
#
# # 模型加载
# model = Classifier()
# model.load_models()
#
# # A榜数据预测
# A_pred = model.predict_voting(A_features)
# A_data['dom'] = A_pred
# A_data.drop('tokenized',axis = 1, inplace = True)
# A_data.to_csv("A_data_Stacking.txt",sep="\t",index=False)

# ============================== 预测A榜数据 dolphin ====================================================
data = pd.read_csv("/mnt/disk/wjh23/EaseDine/ASR/FireRedASR/A_audio_results/FireRed_A_audio_beam_size_5.txt",sep="\t",header=None,names=['uuid','text'])
A_data = process_data(data, drop_duplicates = False)
print(A_data.shape)

A_em = Embeding(A_data)
# Word3Vec
DIM = 100
A_word2vec_embeding = A_em.get_word2vec(f"{str(DOM_path)}/embeding_models/word2vec/word2vec_model_0_{DIM}.bin")

# LDA 测试
num_topics = 20
lda_model_path = f"{str(DOM_path)}/embeding_models/lda/lda_model_{num_topics}.model"
lda_dictionary_path = f"{str(DOM_path)}/embeding_models/lda/lda_dictionary_{num_topics}.gensim"
lda_corpus_path = f"{str(DOM_path)}/embeding_models/lda/lda_corpus_{num_topics}.mm"
A_lda_embeding = A_em.get_lda(num_topics,lda_model_path,lda_dictionary_path,lda_corpus_path)

# 按行拼接（沿 axis=1 水平拼接）
A_features = np.hstack([A_lda_embeding, A_word2vec_embeding])
print("A_features:", A_features.shape)

# 模型加载
model = Classifier()
model.load_models()

# A榜数据预测
A_pred = model.predict_voting(A_features)
A_data['dom'] = A_pred
A_data.drop('tokenized',axis = 1, inplace = True)
# A_data.to_csv("A_data_Stacking.txt",sep="\t",index=False)
print(A_data.head())
# ======================= 补充关键词判断类别 ====================================
from DOM.ML.utils import based_on_keywords

# data = pd.read_csv(r"E:\githubworkspace\EaseDine\DOM\Stacking\A_data_Stacking.txt",sep="\t")

key_df = based_on_keywords(A_data)
key_df.loc[key_df['dom_key']==-1,'dom_key'] = A_data[key_df['dom_key']==-1]['dom']

# 查看预测和关键词判断不同的数据
print("\n查看预测和关键词判断不同的数据：")
temp = key_df[key_df['dom']!=key_df['dom_key']]
print(temp[['text','dom','dom_key']])

key_df['dom'] = key_df['dom_key']
key_df.drop('dom_key',axis=1,inplace=True)
key_df.to_csv("/mnt/disk/wjh23/EaseDine/DOM/A_audio_results/A_audio_recognition_dom_4_14.txt",sep="\t",index=False)
# print(f"联合key准确率:{sum(key_df['dom_key'] == key_df['dom'])/key_df.shape[0]:.4f}")
