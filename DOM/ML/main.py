from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from DOM.NN.utils import loadfile
from train_evaluate import train_predict_evaluate_model
import joblib # 保存和加载模型

#读取数据 (64621,)
combineds, labels = loadfile()
# print(combineds[:2],labels)

# 对数据进行划分
train_corpus, test_corpus, train_labels, test_labels = train_test_split(combineds, labels, test_size=0.2, random_state=0)

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