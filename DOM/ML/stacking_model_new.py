from utils import prepare_data, get_base_models, get_meta_model, batch_predict, get_vectorizers, generate_stratified_meta_features, save_model, evaluate_stacking_model
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

# 生成分层的元特征
X_meta_train, X_meta_test = generate_stratified_meta_features(
    base_models, 
    X_train,
    y_train,
    X_test
)

# 验证数据形状
print(f"元训练集形状: {X_meta_train.shape}, 元测试集形状: {X_meta_test.shape}")

# 训练元模型
meta_model = get_meta_model()
meta_model.fit(X_meta_train, y_train)

# 评估结果
print("\n评估结果:")
print("训练集准确率:", meta_model.score(X_meta_train, y_train))
print("测试集准确率:", meta_model.score(X_meta_test, y_test))
print("\n测试集分类报告:")
print(classification_report(y_test, meta_model.predict(X_meta_test)))

print(f"总耗时: {(time.time() - start_time):.1f} 秒")

# 保存模型
for model_name, model in base_models:
    model_path = f"{root_dir}/DOM/ML/model/{model_name}.pkl"
    save_model(model, model_path)
    
    # 保存特征提取器的词汇表
    vec = model.named_steps[model.steps[0][0]]  # 获取特征提取器
    if hasattr(vec, 'vocabulary_'):
        vocab_path = f"{root_dir}/DOM/ML/model/{model_name}_vocab.pkl"
        joblib.dump(vec.vocabulary_, vocab_path)

# 保存元模型
meta_path = f"{root_dir}/DOM/ML/model/meta_model.pkl"
save_model(meta_model, meta_path)

print("所有模型保存完成。")