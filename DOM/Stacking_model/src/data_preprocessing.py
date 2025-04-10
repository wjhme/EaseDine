import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump

def preprocess_data():
    # 读取数据
    df = pd.read_csv('/mnt/disk/wjh23/EaseDineDatasets/智慧养老_label/train.txt',sep="\t")
    
    # 清洗文本
    df['clean_text'] = df['text'].str.replace(r'天猫精灵', '', regex=True).str.strip()
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['dom'], test_size=0.2, stratify=df['dom'], random_state=42
    )
    
    # TF-IDF特征提取
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
        # stop_words=['我要', '来份', '要份']
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # 保存预处理对象
    dump(tfidf, '../models/tfidf_vectorizer.joblib')
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test