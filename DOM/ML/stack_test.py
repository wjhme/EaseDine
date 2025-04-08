from sklearn.metrics import classification_report
from utils import load_model
import pandas as pd
from pathlib import Path
import numpy as np
import time

# 初始化路径
cur_dir = Path(__file__).parent
root_dir = cur_dir.parent.parent.parent

# 加载测试数据
test_file_path = f"{root_dir}/EaseDineDatasets/pred_A_audio.txt"
data = pd.read_csv(test_file_path, sep="\t", header=None, names=["uttid", "text"])

# 提取测试文本
X_test = data["text"].tolist()

# 配置参数
MODEL_PATH = f"{root_dir}/EaseDine/DOM/ML/model/stacking_model.pkl"

def test_model(model_dict, X_test):
    """多阶段预测的完整测试"""
    base_models = model_dict['base_models']
    meta_model = model_dict['meta_model']


    # 阶段一：基模型预测
    # print("\n[基模型预测]".ljust(50, '-'))
    X_meta = []
    for i, text in enumerate(X_test):
        sample_features = []
        for name, model in base_models:
            try:
                proba = model.predict_proba([text])[0][1]
            except Exception as e:
                print(f"{name} 预测异常: {str(e)}")
                proba = 0.5
            sample_features.append(proba)
            # if i == 0:  # 仅打印第一个样本的基模型输出
            #     print(f"{name:15} | 概率: {proba:.2%}")
        X_meta.append(sample_features)

    # 阶段二：元模型预测
    print("\n[元模型预测]".ljust(50, '-'))
    X_meta = np.array(X_meta)
    pred_labels = meta_model.predict(X_meta)
    return pred_labels



if __name__ == "__main__":
    # 加载模型
    t0 = time.time()
    model = load_model(MODEL_PATH)
    print(f"模型加载时间：{time.time() - t0:.2f} s")

    if model:
        # 执行测试
        t1 = time.time()
        pred_labels = test_model(model, X_test)
        print(f"预测时间：{time.time() - t1:.2f} s")

    # 将结果添加到原始数据中
    data['dom'] = pred_labels

    data.to_csv(f"{root_dir}/EaseDineDatasets/pred_A_audio_with_dom.txt", sep="\t", header=None, index=None)




    '''
    ✅ 模型加载成功
模型加载时间：41.32 s

[有效样本测试]-----------------------------------------
样本1:
  内容: 尊敬的顾客，您的订单#1234已确认，预计送达时间18:30！...
  真实: 正常邮件
  预测: 垃圾邮件 | 置信度: 96.51%
--------------------------------------------------
样本2:
  内容: 【EaseDine】您的预订已成功：3月15日19:00，4人桌。如需修改请致电400-123-45...
  真实: 正常邮件
  预测: 垃圾邮件 | 置信度: 92.55%
--------------------------------------------------
样本3:
  内容: 最后清仓！全场1折起！点击 http://malicious.link 立即抢购！...
  真实: 垃圾邮件
  预测: 垃圾邮件 | 置信度: 99.96%
--------------------------------------------------
样本4:
  内容: 恭喜您获得100元优惠券！立即登录 www.fake-easedine.com 兑换>>...
  真实: 垃圾邮件
  预测: 垃圾邮件 | 置信度: 99.94%
--------------------------------------------------

[异常样本测试]-----------------------------------------
异常样本1: '...' → 预测为: 垃圾邮件
异常样本2: '   ...' → 预测为: 垃圾邮件
异常样本3: '12345612345612345612...' → 预测为: 垃圾邮件
测试时间：0.38 s
    '''