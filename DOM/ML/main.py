from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from DOM.NN.utils import loadfile
from train_evaluate import train_predict_evaluate_model
from sklearn.utils import shuffle
import joblib # 保存和加载模型
import numpy as np
import pandas as pd
import time

t0 = time.time()
#读取数据 (64621,)
combineds, labels = loadfile()
print(f"数据加载完毕.用时{time.time()-t0:.2f}s")

# # 将数据保存为CSV
# data = {
#     'email':combineds,
#     'label':labels
# }
# df = pd.DataFrame(data)
# df.to_csv('E:/githubworkspace/EaseDine/DOM/NN/rawData/data.csv')

# print(labels[:10])
t1 = time.time()
# 打乱数据顺序（生成随机索引）
indices = np.arange(len(combineds))
shuffled_indices = shuffle(indices, random_state=42)

# 定义批次大小（根据内存调整）
batch_size = 1000  # 每批处理1000条数据
test_size = 0.2  # 测试集比例

# 初始化训练集和测试集
train_corpus, test_corpus = [], []
train_labels, test_labels = [], []

# 按批次处理数据
for i in range(0, len(shuffled_indices), batch_size):
    batch_indices = shuffled_indices[i:i + batch_size]

    # 获取当前批次的数据
    batch_texts = [combineds[idx] for idx in batch_indices]
    batch_labels = [labels[idx] for idx in batch_indices]

    # 划分当前批次的训练集和测试集
    split = int(len(batch_texts) * (1 - test_size))
    train_corpus.extend(batch_texts[:split])
    test_corpus.extend(batch_texts[split:])
    train_labels.extend(batch_labels[:split])
    test_labels.extend(batch_labels[split:])
num_train = len(train_corpus)
print(f"训练集大小: {num_train}")
print(f"测试集大小: {len(test_corpus)}")
print(f"数据打乱顺序.用时{time.time()-t1:.2f}s")
#  统计垃圾邮件和正常邮件数量
num_spam = 0
num_ham = 0
for i in range(num_train):
    if train_labels[i] == 0:  #垃圾邮件为0
        num_spam += 1
    elif train_labels[i] == 1:
        num_ham += 1
print(f"spam={num_spam},ham={num_ham}")

def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix

def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

# 构建词向量
# 词袋模型特征
bow_vectorizer, bow_train_features = bow_extractor(train_corpus)
bow_test_features = bow_vectorizer.transform(test_corpus)

# tfidf 特征
tfidf_vectorizer, tfidf_train_features = tfidf_extractor(train_corpus)
tfidf_test_features = tfidf_vectorizer.transform(test_corpus)

# ================ 分别训练贝叶斯分类器、逻辑回归分类器、支持向量机分类器，验证效果 ================================
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

mnb = MultinomialNB()
svm = SGDClassifier(loss='hinge', n_iter_no_change=100)
lr = LogisticRegression()

# 基于词袋模型的多项朴素贝叶斯
print("基于词袋模型特征的贝叶斯分类器:")
mnb_bow_predictions = train_predict_evaluate_model(classifier=mnb,
                                                   train_features=bow_train_features,
                                                   train_labels=train_labels,
                                                   test_features=bow_test_features,
                                                   test_labels=test_labels)
joblib.dump(mnb, 'model/mnb_bow.pkl')

# 基于词袋模型特征的逻辑回归
print("基于词袋模型特征的逻辑回归:")
lr_bow_predictions = train_predict_evaluate_model(classifier=lr,
                                                  train_features=bow_train_features,
                                                  train_labels=train_labels,
                                                  test_features=bow_test_features,
                                                  test_labels=test_labels)
joblib.dump(lr, 'model/lr_bow.pkl')

# 基于词袋模型的支持向量机方法
print("基于词袋模型的支持向量机:")
svm_bow_predictions = train_predict_evaluate_model(classifier=svm,
                                                   train_features=bow_train_features,
                                                   train_labels=train_labels,
                                                   test_features=bow_test_features,
                                                   test_labels=test_labels)
joblib.dump(svm, 'model/svm_bow.pkl')

# 基于tfidf的多项式朴素贝叶斯模型
print("基于tfidf的贝叶斯模型:")
mnb_tfidf_predictions = train_predict_evaluate_model(classifier=mnb,
                                                     train_features=tfidf_train_features,
                                                     train_labels=train_labels,
                                                     test_features=tfidf_test_features,
                                                     test_labels=test_labels)
joblib.dump(mnb, 'model/mnb_tfidf.pkl')

# 基于tfidf的逻辑回归模型
print("基于tfidf的逻辑回归模型:")
lr_tfidf_predictions = train_predict_evaluate_model(classifier=lr,
                                                    train_features=tfidf_train_features,
                                                    train_labels=train_labels,
                                                    test_features=tfidf_test_features,
                                                    test_labels=test_labels)
joblib.dump(lr, 'model/lr_tfidf.pkl')

# 基于tfidf的支持向量机模型
print("基于tfidf的支持向量机模型:")
svm_tfidf_predictions = train_predict_evaluate_model(classifier=svm,
                                                     train_features=tfidf_train_features,
                                                     train_labels=train_labels,
                                                     test_features=tfidf_test_features,
                                                     test_labels=test_labels)
joblib.dump(svm, 'model/svm_tfidf.pkl')