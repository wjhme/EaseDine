from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from sklearn.base import clone
from sklearn.pipeline import Pipeline
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
        # stratify=df_shuffled['dom'],
        random_state=42      # 随机种子
    )

    X_train, X_test = train_df['text'].tolist(), test_df['text'].tolist()
    y_train, y_test = train_df['dom'].tolist(), test_df['dom'].tolist()
    # 数据集统计
    print("\n数据集信息:")
    print(f"训练样本: {len(X_train)} | 测试样本: {len(X_test)}")
    print(f"正样本比例 - 训练集: {np.mean(y_train):.2%} | 测试集: {np.mean(y_test):.2%}")

    return X_train, X_test, y_train, y_test


# -------------------- 特征工程配置 --------------------

# 预测使用:生成分层的元特征
def pred_generate_stratified_meta_features(root_dir, base_models, X_test):
    n_models = len(base_models)
    meta_test = np.zeros((len(X_test), n_models))
    
    for model_idx, (_, model) in enumerate(base_models):
        # 使用特征提取器转换测试数据
        transformer = model.named_steps['countvectorizer'] if 'countvectorizer' in model.named_steps else \
                      model.named_steps['tfidfvectorizer']
        
        # 加载特征提取器的词汇表
        vec_name = list(transformer.named_steps.keys())[0]
        transformer_path = f"{root_dir}/DOM/ML/model/{vec_name}_vocab.pkl"
        transformer.vocabulary_ = joblib.load(transformer_path)
        
        X_transformed = transformer.transform(X_test)
        
        # 进行预测
        final_step = model.steps[-1][1]
        meta_test[:, model_idx] = batch_predict(final_step, X_transformed)
    
    return meta_test

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
    for vec_name, vec in vectorizers:
        models.append((
            f"{vec_name}_nb",
            make_pipeline(vec, MultinomialNB(alpha=0.5))
        ))
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

    models.extend([
        ('rf', make_pipeline(
            TfidfVectorizer(max_features=2000),
            RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=2)
        )),
        ('sgd', make_pipeline(
            TfidfVectorizer(max_features=2000),
            SGDClassifier(loss='log_loss', penalty='elasticnet', max_iter=1000, n_jobs=2)
        ))
    ])
    return models

# -------------------- 元特征生成 --------------------
def generate_stratified_meta_features(base_models, X_train, y_train, X_test):
    """正确的分层元特征生成实现"""
    n_models = len(base_models)
    meta_train = np.zeros((len(X_train), n_models))
    meta_test = np.zeros((len(X_test), n_models))
    
    for model_idx, (name, model) in enumerate(base_models):
        print(f"\n处理基模型 {name} ({model_idx+1}/{n_models})")
        meta_train_col = np.zeros(len(X_train))
        skf = StratifiedKFold(n_splits=n_models, shuffle=True, random_state=42)
        
        # 生成训练集元特征
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr = [X_train[i] for i in train_idx]
            X_val = [X_train[i] for i in val_idx]
            y_tr = [y_train[i] for i in train_idx]
            
            cloned_model = clone(model)
            cloned_model.fit(X_tr, y_tr)
            preds = cloned_model.predict_proba(X_val)[:, 1]
            meta_train_col[val_idx] = preds
        
        # 生成测试集元特征
        full_model = clone(model)
        full_model.fit(X_train, y_train)
        meta_test[:, model_idx] = full_model.predict_proba(X_test)[:, 1]
        meta_train[:, model_idx] = meta_train_col
    
    return meta_train, meta_test

# -------------------- 元模型配置与交叉验证 --------------------
def get_meta_model():
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        early_stopping=True,
        random_state=42
    )
def evaluate_stacking_model(base_models, X_train, y_train):
    """使用交叉验证评估堆叠模型性能"""
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=get_meta_model(),
        cv=KFold(n_splits=5, shuffle=True, random_state=42)  # 定义交叉验证策略
    )
    
    # 计算交叉验证得分
    scores = cross_val_score(stacking_clf, X_train, y_train, cv=5, scoring='accuracy')
    print(f"交叉验证准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return stacking_clf

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

# 加载基模型和元模型
def load_models(root_path):
    """优化后的模型加载函数"""
    # 基模型列表应与训练时完全一致
    base_model_names = [
        'bow_nb', 'bow_svm',
        'tfidf_nb', 'tfidf_svm',
        'rf', 'sgd'
    ]
    
    loaded_models = []
    for name in base_model_names:
        try:
            # 加载完整模型
            model_path = f"{root_path}/EaseDine/DOM/ML/model/{name}.pkl"
            model = joblib.load(model_path)
            
            # 确保特征提取器参数正确
            if isinstance(model, Pipeline):
                vec_name = model.steps[0][0]
                vectorizer = model.named_steps[vec_name]
                
                # 加载对应的词汇表（根据特征提取器类型）
                if 'countvectorizer' in vec_name.lower():
                    vocab_type = 'bow'
                elif 'tfidfvectorizer' in vec_name.lower():
                    vocab_type = 'tfidf'
                
                vocab_path = f"{root_path}/EaseDine/DOM/ML/model/{vocab_type}_vocab.pkl"
                vectorizer.vocabulary_ = joblib.load(vocab_path)
                vectorizer.fixed_vocabulary_ = True  # 关键修复点
                
            loaded_models.append((name, model))
        except Exception as e:
            print(f"⚠️ 加载模型 {name} 失败: {str(e)}")
            continue
    
    # 加载元模型
    try:
        meta_model = joblib.load(f"{root_path}/EaseDine/DOM/ML/model/meta_model.pkl")
    except Exception as e:
        print(f"❌ 加载元模型失败: {str(e)}")
        raise
    
    print(f"✅ 成功加载 {len(loaded_models)} 个基模型和元模型")
    return loaded_models, meta_model

def generate_meta_features(models, texts):
    """改进的元特征生成"""
    meta_features = np.zeros((len(texts), len(models)))
    
    for idx, (name, model) in enumerate(models):
        try:
            # 确保使用完整的pipeline
            if isinstance(model, Pipeline):
                processed_texts = model.steps[0][1].transform(texts)
                final_model = model.steps[-1][1]
                meta_features[:, idx] = batch_predict(final_model, processed_texts)
            else:
                meta_features[:, idx] = batch_predict(model, texts)
        except Exception as e:
            print(f"❌ 模型 {name} 特征生成失败: {str(e)}")
            meta_features[:, idx] = 0.5  # 中性值填充
    
    return meta_features

def save_model(model, path):
    """安全保存模型"""
    try:
        joblib.dump(model, path)
        print(f"✅ 模型已成功保存到 {path}")
    except Exception as e:
        print(f"❌ 模型保存失败: {str(e)}")

def load_model(path):
    """安全加载模型"""
    try:
        model = joblib.load(path)
        print("✅ 模型加载成功")
        return model
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return None


if __name__=="__main__":
    # 加载测试数据
    test_file_path = f"/mnt/disk/wjh23/EaseDineDatasets/pred_A_audio.txt"
    data = pd.read_csv(test_file_path, sep="\t", header=None, names=["uuid", "text"])
    data = based_on_keywords(data)
    data.to_csv("/mnt/disk/wjh23/EaseDineDatasets/pred_A_audio_keyword.txt", sep="\t", header=None, index=None)