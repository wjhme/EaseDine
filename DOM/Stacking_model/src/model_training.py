from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from joblib import dump
from data_preprocessing import preprocess_data

def train_stack_model():
    # 获取预处理数据
    X_train, X_test, y_train, y_test = preprocess_data()
    
    # 定义基学习器
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(kernel='linear', probability=True, random_state=42)),
        ('nb', MultinomialNB())
    ]
    
    # 定义元学习器
    meta_model = LogisticRegression(max_iter=1000)
    
    # 构建Stacking模型
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        stack_method='auto',
        n_jobs=-1
    )
    
    # 训练模型
    stacking_model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = stacking_model.predict(X_test)
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # 保存模型
    dump(stacking_model, '../models/stacking_model.joblib')
    print("Model saved successfully.")
    
    return stacking_model

if __name__ == "__main__":
    train_stack_model()