import pandas as pd
from joblib import load
import warnings
warnings.filterwarnings('ignore')

class StackingPredictor:
    def __init__(self):
        self.model = load('../models/stacking_model.joblib')
        self.tfidf = load('../models/tfidf_vectorizer.joblib')
        
    def preprocess_text(self, text):
        return text.replace('天猫精灵', '').strip()
    
    def predict(self, texts):
        # 输入支持字符串或列表
        if isinstance(texts, str):
            texts = [texts]
            
        # 预处理
        clean_texts = [self.preprocess_text(t) for t in texts]
        
        # 特征转换
        tfidf_features = self.tfidf.transform(clean_texts)
        
        # 预测
        probs = self.model.predict_proba(tfidf_features)
        predictions = self.model.predict(tfidf_features)
        
        # 格式化结果
        results = []
        for text, pred, prob in zip(texts, predictions, probs):
            results.append({
                "text": text,
                "prediction": int(pred),
                "probability": round(float(prob[pred]), 4),
                "class_0_prob": round(float(prob[0]), 4),
                "class_1_prob": round(float(prob[1]), 4)
            })
        return results

if __name__ == "__main__":
    # 示例预测
    predictor = StackingPredictor()
    test_samples = [
        "天猫精灵我要吃红烧牛肉面",
        "帮我计算三角函数的值",
        "来份西红柿炒鸡蛋"
    ]
    
    results = predictor.predict(test_samples)
    print("Prediction Results:")
    for res in results:
        print(f"Text: {res['text']}")
        print(f"Prediction: {res['prediction']} (Prob: {res['probability']:.4f})")
        print("-" * 50)