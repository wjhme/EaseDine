import sys
from Classifiers import DOM
from pathlib import Path

# 获取当前文件的绝对路径
current_file = Path(__file__).resolve()
# 获取项目根目录（假设项目根目录是 EaseDine 的父目录）
project_root = current_file.parent.parent.parent  # 根据实际层级调整
DOM_path = current_file.parent.parent
# 将项目根目录添加到 Python 路径
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from DOM.utils import process_data, Embeding
from sklearn.model_selection import train_test_split

data = pd.read_csv("/mnt/disk/wjh23/EaseDineDatasets/智慧养老_label/train.txt",sep="\t")
data = process_data(data)
train_df, test_df = train_test_split(
        data,
        test_size=0.15,       # 测试集比例
        stratify=data['dom'],
        random_state=42      # 随机种子
    )
train_label = train_df['dom'].tolist()
test_label = test_df['dom'].tolist()
print(train_df.shape)
print(len(train_label))

print(test_df.shape)
print(len(test_label))

# train_em = Embeding(train_df)
# Word3Vec 测试
DIM = 100
# train_em.train_word2vec(DIM)
# train_word2vec_embeding = train_em.get_word2vec(f"{str(DOM_path)}/embeding_models/word2vec/word2vec_model_0_{DIM}.bin")

# LDA 测试
num_topics = 20
# train_em.train_lda(num_topics = num_topics)
lda_model_path = f"{str(DOM_path)}/embeding_models/lda/lda_model_{num_topics}.model"
lda_dictionary_path = f"{str(DOM_path)}/embeding_models/lda/lda_dictionary_{num_topics}.gensim"
lda_corpus_path = f"{str(DOM_path)}/embeding_models/lda/lda_corpus_{num_topics}.mm"
# train_lda_embeding = train_em.get_lda(num_topics,lda_model_path,lda_dictionary_path,lda_corpus_path)

# print("LDA 嵌入形状:", train_lda_embeding.shape)
# print("Word2Vec 嵌入形状:", train_word2vec_embeding.shape)

# # 按行拼接（沿 axis=1 水平拼接）
# train_features = np.hstack([train_lda_embeding, train_word2vec_embeding])
# print("train_features形状:", train_features.shape)


test_em = Embeding(test_df)
test_word2vec_embeding = test_em.get_word2vec(f"{str(DOM_path)}/embeding_models/word2vec/word2vec_model_0_{DIM}.bin")
test_lda_embeding = test_em.get_lda(num_topics,lda_model_path,lda_dictionary_path,lda_corpus_path)

# 按行拼接（沿 axis=1 水平拼接）
test_features = np.hstack([test_lda_embeding, test_word2vec_embeding])
print("test_features形状:", test_features.shape)

# 初始化并训练
model = DOM()
# model.train(train_features, train_label)

# train_pred = model.predict_voting(test_features)
# acc = model.evaluate(train_pred, test_label)
# print(f"训练集准确率: {acc:.4f}")

# 加载模型并预测
model.load_models()
test_pred = model.predict_voting(test_features)
acc = model.evaluate(test_pred, test_label)
test_df['dom_pred'] =test_pred
test_df.drop('tokenized',axis = 1, inplace = True)
print(f"测试集准确率: {acc:.4f}")

from DOM.ML.utils import based_on_keywords
key_df = based_on_keywords(test_df)
key_df.loc[key_df['dom_key']==-1,'dom_key'] = test_df[key_df['dom_key']==-1]['dom_pred']
key_df.to_csv("test_data_Stacking_key.txt",sep="\t",index=False)
print(f"联合key准确率:{sum(key_df['dom_key'] == key_df['dom'])/key_df.shape[0]:.4f}")