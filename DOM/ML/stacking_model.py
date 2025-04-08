from utils import prepare_data, get_base_models, get_meta_model, batch_predict, get_vectorizers
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import joblib
import time
import sys
from pathlib import Path
cur_dir = Path(__file__).parent
root_dir = cur_dir.parent.parent

# 初始化计时
start_time = time.time()

# 数据准备
file_dir = f"/mnt/disk/wjh23/EaseDineDatasets/智慧养老_label/train.txt"
X_train, X_test, y_train, y_test = prepare_data(file_dir)

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
print(f"总耗时: {(time.time() - start_time):.1f} 秒")

# 保存模型
# 修改后的保存代码
stacking_model = {
    'base_models': base_models,  # 保存所有基模型
    'meta_model': meta_model     # 保存元模型
}
joblib.dump(stacking_model, f'{root_dir}/DOM/ML/model/stacking_model.pkl')
