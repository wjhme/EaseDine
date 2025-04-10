import os
import numpy as np
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class Classifier:
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
            
    def load_models(self):
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
            self.load_models()
            
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
            self.load_models()
        return self.trained_models[model_name].predict(X_test)
    
    def evaluate(self, pred_label, true_label):
        """
        评估模型
        评价指标:准确率
        """
        acc = accuracy_score(true_label, pred_label)

        return acc
    
if __name__ == "__main__":
    # 初始化并训练
    ensemble = Classifier()
    ensemble.train(X_train, y_train)

    # 加载测试数据
    X_test = np.load("test_features.npy")

    # 加载模型并预测
    ensemble.load_models()
    y_pred = ensemble.predict_voting(X_test)

    # 查看单个模型预测
    y_pred_svm = ensemble.predict_single(X_test, "svm")