import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from pathlib import Path
# 获取当前文件的绝对路径
current_file = Path(__file__).resolve()
# 获取项目根目录（假设项目根目录是 EaseDine 的父目录）
project_root = current_file.parent.parent.parent  # 根据实际层级调整
DOM_path = current_file.parent.parent
# 将项目根目录添加到 Python 路径
sys.path.append(str(project_root))
from DOM.utils import process_data, based_on_keywords, Embeding

class DOM:
    def __init__(self, model_dir="saved_models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 初始化基分类器
        self.classifiers = {
            "naive_bayes": GaussianNB(),
            "logistic_regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
            "svm": SVC(probability=True),
            "random_forest": RandomForestClassifier()
        }
        
    def train(self, X_train, y_train):
        """训练所有基分类器并自动保存"""
        for name, clf in self.classifiers.items():
            clf.fit(X_train, y_train)
            joblib.dump(clf, os.path.join(self.model_dir, f"{name}.pkl"))
            
    def _load_models(self):
        """加载已训练的模型"""
        self.trained_models = {}
        for name in self.classifiers.keys():
            path = os.path.join(self.model_dir, f"{name}.pkl")
            if os.path.exists(path):
                self.trained_models[name] = joblib.load(path)
            else:
                raise FileNotFoundError(f"Model {name} not found at {path}")
    
    def predict_voting(self, X_test):
        """集成预测（硬投票）"""
        if not hasattr(self, 'trained_models'):
            self._load_models()
            
        predictions = []
        for name, model in self.trained_models.items():
            pred = model.predict(X_test)
            predictions.append(pred)
        
        # 多数投票
        final_pred = []
        for sample_preds in np.array(predictions).T:
            values, counts = np.unique(sample_preds, return_counts=True)
            final_pred.append(values[np.argmax(counts)])
            
        return np.array(final_pred)
    
    def predict_single(self, X_test, model_name):
        """单个基分类器预测"""
        if not hasattr(self, 'trained_models'):
            self._load_models()
        return self.trained_models[model_name].predict(X_test)
    
    def pre_dom(self, recognized_path, save_path):
        '''根据识别的文本预测DOM类别'''

        # 读取数据
        recognized_data = pd.read_csv(recognized_path,sep="\t",header=None,names=['uuid','text'])
        # 数据预处理
        processed_data = process_data(recognized_data, drop_duplicates = False)

        # 生成词向量
        em = Embeding(processed_data)
        # Word3Vec
        DIM = 100
        word2vec_embeding = em.get_word2vec(f"{str(DOM_path)}/embeding_models/word2vec/word2vec_model_0_{DIM}.bin")

        # LDA 主题特征
        num_topics = 20
        lda_model_path = f"{str(DOM_path)}/embeding_models/lda/lda_model_{num_topics}.model"
        lda_dictionary_path = f"{str(DOM_path)}/embeding_models/lda/lda_dictionary_{num_topics}.gensim"
        lda_corpus_path = f"{str(DOM_path)}/embeding_models/lda/lda_corpus_{num_topics}.mm"
        lda_embeding = em.get_lda(num_topics,lda_model_path,lda_dictionary_path,lda_corpus_path)

        # 按行拼接（沿 axis=1 水平拼接）
        features = np.hstack([lda_embeding, word2vec_embeding])
        print("生成的数据特征形状:", features.shape)

        # 类别预测
        pred = self.predict_voting(features)
        processed_data['dom'] = pred
        processed_data.drop('tokenized',axis = 1, inplace = True)

        # ======================= 关键词判断类别 进行修正 ====================================
        key_df = based_on_keywords(processed_data)
        key_df.loc[key_df['dom_key']==-1,'dom_key'] = processed_data[key_df['dom_key']==-1]['dom']

        # 查看预测和关键词判断不同的数据
        print("\n查看预测和关键词判断不同的数据：")
        temp = key_df[key_df['dom']!=key_df['dom_key']]
        print(temp[['uuid','text','dom','dom_key']])

        key_df['dom'] = key_df['dom_key']
        key_df.drop('dom_key',axis=1,inplace=True)

        # 保存结果
        key_df.to_csv(save_path, sep="\t", index=False)
    
    def pre_dom_str(self, recognized_str):
        '''根据识别的文本预测DOM类别'''

        # 数据预处理
        processed_data = process_data(recognized_str, drop_duplicates = False)

        # 生成词向量
        em = Embeding(processed_data)
        # Word3Vec
        DIM = 100
        word2vec_embeding = em.get_word2vec(f"{str(DOM_path)}/embeding_models/word2vec/word2vec_model_0_{DIM}.bin")

        # LDA 主题特征
        num_topics = 20
        lda_model_path = f"{str(DOM_path)}/embeding_models/lda/lda_model_{num_topics}.model"
        lda_dictionary_path = f"{str(DOM_path)}/embeding_models/lda/lda_dictionary_{num_topics}.gensim"
        lda_corpus_path = f"{str(DOM_path)}/embeding_models/lda/lda_corpus_{num_topics}.mm"
        lda_embeding = em.get_lda(num_topics,lda_model_path,lda_dictionary_path,lda_corpus_path)

        # 按行拼接（沿 axis=1 水平拼接）
        features = np.hstack([lda_embeding, word2vec_embeding])
        print("生成的数据特征形状:", features.shape)

        # 类别预测
        pred = self.predict_voting(features)

        return pred
    
    def evaluate(self, pred_label, true_label):
        """
        评估模型
        评价指标:准确率
        """
        acc = accuracy_score(true_label, pred_label)

        return acc
    
if __name__ == "__main__":
    # 初始化并训练
    ensemble = DOM()
    ensemble.train(X_train, y_train)

    # 加载测试数据
    X_test = np.load("test_features.npy")

    # 加载模型并预测
    ensemble.load_models()
    y_pred = ensemble.predict_voting(X_test)

    # 查看单个模型预测
    y_pred_svm = ensemble.predict_single(X_test, "svm")