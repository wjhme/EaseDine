from sklearn.metrics import classification_report
from utils import load_model
import numpy as np
import time

# 配置参数
MODEL_PATH = r"E:\githubworkspace\EaseDine\DOM\ML\model\stacking\stacking_model.pkl"
TEST_EXAMPLES = [
    # 正常邮件样例
    ("尊敬的顾客，您的订单#1234已确认，预计送达时间18:30！", 1),
    ("【EaseDine】您的预订已成功：3月15日19:00，4人桌。如需修改请致电400-123-4567", 1),

    # 垃圾邮件样例
    ("最后清仓！全场1折起！点击 http://malicious.link 立即抢购！", 0),
    ("恭喜您获得100元优惠券！立即登录 www.fake-easedine.com 兑换>>", 0),

    # 边界测试样例
    ("", -1),  # 空文本
    ("   ", -1),  # 纯空格
    ("123456" * 500, -1)  # 超长数字文本
]

def test_model(model_dict, examples):
    """多阶段预测的完整测试"""
    base_models = model_dict['base_models']
    meta_model = model_dict['meta_model']

    # 分离样本
    valid_samples = [(t, l) for t, l in examples if l != -1]
    invalid_samples = [t for t, l in examples if l == -1]

    if valid_samples:
        texts, true_labels = zip(*valid_samples)

        # 阶段一：基模型预测
        # print("\n[基模型预测]".ljust(50, '-'))
        X_meta = []
        for i, text in enumerate(texts):
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
        probas = meta_model.predict_proba(X_meta)

        # 展示结果
        print("\n[有效样本测试]".ljust(50, '-'))
        for i, (text, true, pred) in enumerate(zip(texts, true_labels, pred_labels)):
            conf = probas[i][1] if pred == 1 else probas[i][0]
            print(f"样本{i + 1}:")
            print(f"  内容: {text[:50]}...")  # 显示前50字符
            print(f"  真实: {'正常邮件' if true else '垃圾邮件'}")
            print(f"  预测: {'正常邮件' if pred else '垃圾邮件'} | 置信度: {conf:.2%}")
            print("-" * 50)

    # 测试异常样本
    if invalid_samples:
        print("\n[异常样本测试]".ljust(50, '-'))
        for i, text in enumerate(invalid_samples):
            try:
                pred = model.predict([text])[0]
                print(f"异常样本{i + 1}: '{text[:20]}...' → 预测为: {'正常邮件' if pred else '垃圾邮件'}")
            except Exception as e:
                print(f"异常样本{i + 1} 预测失败: {str(e)}")


if __name__ == "__main__":
    # 加载模型
    t0 = time.time()
    model = load_model(MODEL_PATH)
    print(f"模型加载时间：{time.time() - t0:.2f} s")

    if model:
        # 执行测试
        t1 = time.time()
        test_model(model, TEST_EXAMPLES)
        print(f"测试时间：{time.time() - t1:.2f} s")

        # # 补充测试（可选）
        # print("\n[压力测试]".ljust(50, '-'))
        # print("生成1000个随机样本测试...")
        # random_texts = ["Sample text " + str(i) for i in range(1000)]
        # _ = model.predict(random_texts)
        # print("压力测试完成，无异常抛出")


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