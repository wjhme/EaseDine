from utils import prepare_data, get_base_models, get_meta_model, batch_predict, get_vectorizers
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import joblib
import time
import sys

# 初始化计时
start_time = time.time()

# 数据准备
X_train, X_test, y_train, y_test = prepare_data()

# 获取模型组件
vectorizers = get_vectorizers()
base_models = get_base_models(vectorizers)
meta_model = get_meta_model()

# 训练基模型
print("\n训练基模型:")
meta_features = []
base_accuracies = []

for name, model in base_models:
    try:
        # 训练并记录时间
        iter_start = time.time()
        model.fit(X_train, y_train)

        # 收集预测结果
        train_pred = batch_predict(model, X_train)
        test_pred = batch_predict(model, X_test)

        # 存储元特征
        meta_features.append((train_pred, test_pred))

        # 评估基模型
        acc = accuracy_score(y_test, (test_pred > 0.5).astype(int))
        base_accuracies.append((name, acc))
        print(f"{name:15} | 准确率: {acc:.4f} | 耗时: {time.time() - iter_start:.1f}s")

    except Exception as e:
        print(f"{name} 训练失败: {str(e)}")
        continue

# 构建元数据集
X_meta_train = np.vstack([p[0] for p in meta_features]).T
X_meta_test = np.vstack([p[1] for p in meta_features]).T

# 训练元模型
print("\n训练元模型...")
meta_model.fit(X_meta_train, y_train)

# 最终评估
final_pred = meta_model.predict(X_meta_test)
print("\n最终模型评估:")
print(classification_report(y_test, final_pred))
print(f"总耗时: {(time.time() - start_time)/60:.1f} 分")

# 保存模型
# 修改后的保存代码
stacking_model = {
    'base_models': base_models,  # 保存所有基模型
    'meta_model': meta_model     # 保存元模型
}
joblib.dump(stacking_model, 'model/stacking/stacking_model.pkl')


''' 
数据集信息:
训练样本: 51696 | 测试样本: 12925
正样本比例 - 训练集: 33.68% | 测试集: 33.68%
prepare_data 内存使用: 131.68 MB

训练基模型:
bow_nb          | 准确率: 0.9390 | 耗时: 55.4s
bow_svm         | 准确率: 0.9826 | 耗时: 61.6s
tfidf_nb        | 准确率: 0.9515 | 耗时: 54.2s
tfidf_svm       | 准确率: 0.9856 | 耗时: 56.3s
hash_nb         | 准确率: 0.9912 | 耗时: 101.0s
rf              | 准确率: 0.9673 | 耗时: 30.1s
sgd             | 准确率: 0.9793 | 耗时: 22.6s

训练元模型...

最终模型评估:
              precision    recall  f1-score   support

           0       0.99      1.00      0.99      8572
           1       0.99      0.99      0.99      4353

    accuracy                           0.99     12925
   macro avg       0.99      0.99      0.99     12925
weighted avg       0.99      0.99      0.99     12925

总耗时: 7.3 分
'''
