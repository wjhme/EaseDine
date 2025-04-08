from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import joblib
import psutil


# -------------------- 内存监控装饰器 --------------------
def memory_monitor(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_mem = process.memory_info().rss / 1024 ** 2
        result = func(*args, **kwargs)
        end_mem = process.memory_info().rss / 1024 ** 2
        print(f"{func.__name__} 内存使用: {end_mem - start_mem:.2f} MB")
        return result

    return wrapper


# -------------------- 数据准备 --------------------
@memory_monitor
def prepare_data(file_dir):
    """加载并预处理数据"""
    df = pd.read_csv(file_dir, sep="\t")
    df_shuffled = df.sample(frac=1, random_state=42)
    # 文本清洗（简单示例）
    # cleaned_texts = [text.lower().replace('\n', ' ') for text in texts]

    # 打乱并分割数据集
    train_df, test_df = train_test_split(
        df_shuffled, 
        test_size=0.2,       # 测试集比例
        # stratify=labels,
        random_state=42      # 随机种子
    )

    X_train, X_test, y_train, y_test = train_df['text'].tolist(), test_df['text'].tolist(), train_df['dom'].tolist(), test_df['dom'].tolist()
    # 数据集统计
    print("\n数据集信息:")
    print(f"训练样本: {len(X_train)} | 测试样本: {len(X_test)}")
    print(f"正样本比例 - 训练集: {np.mean(y_train):.2%} | 测试集: {np.mean(y_test):.2%}")

    return X_train, X_test, y_train, y_test


# -------------------- 特征工程配置 --------------------
def get_vectorizers():
    """返回不同特征提取器"""
    common_params = {
        'ngram_range': (1, 2),
        'min_df': 5,
        'max_features': 1000,
        'dtype': np.float32
    }

    return [
        ('bow', CountVectorizer(binary=True, **common_params)),
        ('tfidf', TfidfVectorizer(sublinear_tf=True, **common_params)),
        ('hash', TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            use_idf=False,
            norm=None
        ))
    ]


# -------------------- 基模型配置 --------------------
def get_base_models(vectorizers):
    """创建多样化的基模型"""
    models = []

    # 基于不同特征提取器的模型
    for vec_name, vec in vectorizers:
        # 始终添加朴素贝叶斯模型
        models.append((
            f"{vec_name}_nb",
            make_pipeline(
                vec,
                MultinomialNB(alpha=0.5)
            )
        ))

        # 仅对非hash特征提取器添加SVM模型
        if vec_name != 'hash':
            models.append((
                f"{vec_name}_svm",
                make_pipeline(
                    vec,
                    CalibratedClassifierCV(
                        LinearSVC(C=0.5, dual=False, max_iter=1000),
                        method='sigmoid',
                        cv=3
                    )
                )
            ))

    # 添加其他类型模型
    models.extend([
        ('rf', make_pipeline(
            TfidfVectorizer(max_features=2000),
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                n_jobs=2
            )
        )),
        ('sgd', make_pipeline(
            TfidfVectorizer(max_features=2000),
            SGDClassifier(
                loss='log_loss',
                penalty='elasticnet',
                max_iter=1000,
                n_jobs=2
            )
        ))
    ])

    return models


# -------------------- 元模型配置 --------------------
def get_meta_model():
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        early_stopping=True,
        random_state=42
    )


# -------------------- 分批预测函数 --------------------
def batch_predict(model, X, batch_size=2000):
    """内存友好的分批预测"""
    predictions = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        try:
            preds = model.predict_proba(batch)[:, 1]
        except AttributeError:
            dec = model.decision_function(batch)
            preds = 1 / (1 + np.exp(-dec))
        predictions.append(preds)
    return np.concatenate(predictions)

def load_model(path):
    """安全加载模型"""
    try:
        model = joblib.load(path)
        print("✅ 模型加载成功")
        return model
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return None