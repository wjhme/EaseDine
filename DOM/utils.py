import collections
import jieba  # 中文分词
import torch
import xxhash
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import time
from gensim.models import Word2Vec
from gensim import corpora, models
# from glove import Corpus, Glove
import json

def process_data(data_df, drop_duplicates = True):
    '''
    数据预处理：
    1.分词、去停用词
    2.去重
    
    '''
    t0 = time.time()
    # 分词、去停用词
    data = get_tokenized(data_df)
    print(f"分词、去停用词用时：{time.time() - t0:.2f} s")

    if drop_duplicates:
        # 去重
        # data = data.drop_duplicates(subset=['tokenized'],keep = 'first')

        # 将 tokenized 列转换为哈希值 (提速关键)
        data['tokenized_hash'] = data['tokenized'].apply(
            lambda x: xxhash.xxh64(str(x)).hexdigest()
        )
        
        # 去重（基于哈希值）
        data = data.drop_duplicates(subset=['tokenized_hash'], keep='first')
        data = data.drop(columns=['tokenized_hash'])

        print(f"原始数据大小：{data_df.shape[0]},重复记录有{data_df.shape[0] - data.shape[0]}个,当前数据大小：{data.shape[0]}")
    print(f"数据处理用时：{time.time() - t0:.2f} s")
    return data

class Embeding():
    def __init__(self, data):
        self.data = data

    def train_word2vec(self, vector_size = 100, sg = 0):
        '''
        训练Word2Vec模型
        '''
        # 提取分词后的句子列表
        sentences = self.data['tokenized'].tolist()

        if not sg:
            # 训练 Word2Vec 模型 CBOW 模型(适合快速训练和常见任务，但对低频词敏感)
            model = Word2Vec(
                sentences,
                vector_size=vector_size,    # 向量维度
                window=3,           # 上下文窗口大小
                min_count=1,        # 忽略出现次数低于此值的词
                workers=8,           # 并行线程数
                sg=0
            )
        else:
            # 训练 Word2Vec 模型 Skip-Gram 模型(计算成本高，但能更好地处理低频词和复杂语义)
            model = Word2Vec(
                sentences,
                vector_size=vector_size,    # 向量维度
                window=3,           # 上下文窗口大小
                min_count=1,        # 忽略出现次数低于此值的词
                workers=8,           # 并行线程数
                sg=1
            )
        model.save(f"./EaseDine/DOM/embeding_models/word2vec/word2vec_model_{sg}_{vector_size}.bin")  # 二进制格式（保存模型结构和权重）

    def get_word2vec(self, embeding_model_path, sg=0, norm_type='l2'):
        '''
        生成Word2Vec词向量，可选归一化
        
        参数:
            embeding_model_path: Word2Vec模型路径
            sg: 模型类型（0=CBOW, 1=Skip-Gram）
            norm_type: 归一化类型 ('l2', 'zscore', None)
        
        返回:
            np.ndarray: 形状为 (样本数, 词向量维度)
        '''
        # 加载模型
        model = Word2Vec.load(embeding_model_path)
        word_vectors = model.wv
        vector_size = word_vectors.vector_size

        # 生成句子向量（均值池化）
        def get_sentence_vector(sentence):
            vectors = [word_vectors[word] for word in sentence if word in word_vectors]
            if not vectors:
                return np.zeros(vector_size)
            return np.mean(vectors, axis=0)

        # 获取所有句子的向量（二维数组）
        embeddings = np.stack([get_sentence_vector(sentence) for sentence in self.data['tokenized']])

        # 归一化处理
        if norm_type == 'l2':
            embeddings = normalize(embeddings, norm='l2')  # L2归一化（模长为1）
        elif norm_type == 'zscore':
            scaler = StandardScaler()
            embeddings = scaler.fit_transform(embeddings)  # 标准化（均值为0，方差为1）
        # 若 norm_type=None，保持原始值

        return embeddings  # 形状: (样本数, 词向量维度)

    def train_lda(self, num_topics = 5):
        '''
        训练LDA主题特征提取模型
        '''
        # 提取分词后的句子列表
        sentences = self.data['tokenized'].tolist()

        # 创建词典（词汇表）
        dictionary = corpora.Dictionary(sentences)
        print("原始词典大小:", len(dictionary))

        # 过滤低频词（保留出现至少no_below次的词）
        dictionary.filter_extremes(no_below=2)
        print("过滤后词典大小:", len(dictionary))

        # 转换为文档-词袋向量（稀疏表示）
        corpus = [dictionary.doc2bow(text) for text in sentences]

        # 训练LDA模型
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=20,         # 迭代次数
            alpha='auto',      # 主题稀疏性（自动调整）
            eta='auto'         # 词稀疏性（自动调整）
        )
        # 保存模型
        lda_model.save(f"./EaseDine/DOM/embeding_models/lda/lda_model_{num_topics}.model")
        # 保存词典（Dictionary）
        dictionary.save(f"./EaseDine/DOM/embeding_models/lda/lda_dictionary_{num_topics}.gensim")
        # 保存语料库（可选，MM 格式）
        corpora.MmCorpus.serialize(f"./EaseDine/DOM/embeding_models/lda/lda_corpus_{num_topics}.mm", corpus)

        # 打印每个主题的关键词
        print("\n主题关键词分布:")
        for topic_id in range(num_topics):
            print(f"主题 {topic_id}: {lda_model.print_topic(topic_id)}")

    def get_lda(self, num_topics, lda_model_path, lda_dictionary_path, lda_corpus_path):
        '''
        生成LDA主题特征
        '''
        # 加载模型
        lda_model = models.LdaModel.load(lda_model_path)

        # 加载词典
        dictionary = corpora.Dictionary.load(lda_dictionary_path)

        # 加载语料库
        # corpus = corpora.MmCorpus(lda_corpus_path)

        # 生成语料库
        new_corpus = [dictionary.doc2bow(text) for text in self.data['tokenized']]
        
        # 提取文档-主题分布
        doc_topics = [lda_model.get_document_topics(doc) for doc in new_corpus]

        # 提取文档-主题分布作为特征
        # 转换为稠密矩阵（每行是一个文档的主题概率）
        def get_topic_vector(topics, num_topics):
            vec = np.zeros(num_topics)
            for topic_id, prob in topics:
                vec[topic_id] = prob
            return vec

        lda_embeding = np.array([get_topic_vector(doc, num_topics) for doc in doc_topics])

        # 打印结果
        # print("\n文档-主题特征矩阵:")
        # print(df[['text', 'topic_vector']])
        return lda_embeding

    # def train_glove(self, vector_size = 100):
    #     '''
    #     训练GloVe模型
    #     '''
    #     # 提取分词后的句子列表
    #     sentences = self.data['tokenized'].tolist()

    #     # 步骤1：构建共现矩阵
    #     corpus = Corpus()
    #     corpus.fit(sentences, window=3)  # window=5 表示左右各5个词

    #     # 步骤2：训练 GloVe 模型
    #     glove = Glove(no_components=vector_size, learning_rate=0.05)  # 向量维度=100
    #     glove.fit(corpus.matrix, epochs=30, verbose=True)
    #     glove.add_dictionary(corpus.dictionary)  # 添加词表

    #     # 保存词向量（numpy格式）
    #     np.save(f"./EaseDine/DOM/embeding_models/glove_vectors_{vector_size}.npy", glove.word_vectors)

    #     # 保存词典（词到索引的映射）
    #     with open(f"./EaseDine/DOM/embeding_models/glove_dict_{vector_size}.json", "w") as f:
    #         json.dump(glove.dictionary, f)

    # def get_glove(self, glove_vectors_path, glove_dict_path):
    #     '''
    #     生成GloVe
    #     '''
    #     # 加载词向量和词典
    #     word_vectors = np.load(glove_vectors_path)
    #     with open(glove_dict_path, "r") as f:
    #         dictionary = json.load(f)

    #     # 将整个句子转换为向量（取词向量均值）
    #     def get_glove_sentence_vector(sentence):

    #         indices = [dictionary[word] for word in sentence if word in dictionary]
    #         if not indices:
    #             return None
    #         vectors = word_vectors[indices]
    #         return np.mean(vectors, axis=0)

    #     # 添加句子向量到 DataFrame
    #     glove_embeding = self.data['tokenized'].apply(get_glove_sentence_vector)
    #     # print(df[['text', 'glove_vector']])

    #     return glove_embeding

def get_tokenized(data, stopwords_path = "/mnt/disk/wjh23/EaseDine/DOM/stopwords/my_stopwords.txt"):
    # 复制数据避免修改原数据
    df = data.copy()

    # 加载停用词表
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])

    def tokenize(text):
        # 分词并过滤停用词/空字符/标点
        words = jieba.lcut(str(text))
        return [
            word for word in words
            if (word not in stopwords) and       # 去停用词
               (word.strip() != '') and          # 去空字符
               (not word.isspace()) and          # 去空白符
               (not all(c in '，。！？；：“”‘’（）【】…—' for c in word))  # 去中文标点
        ]

    df['tokenized'] = df['text'].apply(tokenize)
    # print(max(df['text'].apply(lambda x:len(x))))
    return df

class Vocabulary:
    '''词汇表类'''
    def __init__(self, counter, min_freq=1, reserved_tokens=None):
        self.itos = reserved_tokens.copy() if reserved_tokens else []
        self.token_to_idx = {token: i for i, token in enumerate(self.itos)}

        # 添加频率足够的词
        for token, count in counter.items():
            if count >= min_freq and token not in self.token_to_idx:
                self.itos.append(token)
                self.token_to_idx[token] = len(self.itos) - 1

    def __getitem__(self, token):
        return self.token_to_idx.get(token, 0)

    def __len__(self):
        """返回词汇表大小"""
        return len(self.itos)

    def to_indices(self, tokens):
        """将token列表转换为索引列表"""
        return [self[token] for token in tokens]

# 从分词后的数据构建词汇表（Vocabulary），包含低频词过滤和保留特殊符号（如 <pad>）
def get_vocab(tokenized_data):
    # 1. 分词处理
    # tokenized_data = get_tokenized(data)
    # 2. 统计词频
    counter = collections.Counter([tk for st in tokenized_data['tokenized'] for tk in st])
    # 3. 构建词汇表
    return Vocabulary(counter, min_freq = 1, reserved_tokens = ['<pad>','<unk>'])


def tokenized_data(tokenized_data, vocab, max_l=55):
    """
    改进后的预处理函数：
    1. 添加去重逻辑
    2. 验证数据完整性
    """
    # 检查输入
    if 'text' not in tokenized_data.columns or 'dom' not in tokenized_data.columns:
        raise ValueError("输入数据需要包含'text'和'dom'列")

    # 分词和编码
    # tokenized_data = get_tokenized(data)
    features = []
    labels = []

    # 确保每个样本独立处理
    for _, row in tokenized_data.iterrows():
        words = row['text']
        indices = vocab.to_indices(words)
        if len(indices) < max_l:
            padding = [vocab['<pad>']] * (max_l - len(indices))  # 仅用pad
            indices += padding
        else:
            indices = indices[:max_l]
        features.append(indices)
        labels.append(row['dom'])

    # 转换为tensor并去重
    features = torch.tensor(features)
    labels = torch.tensor(labels)

    # 重要！去除完全相同的样本
    unique_features, unique_indices = torch.unique(features, dim=0, return_inverse=True)
    unique_labels = labels[torch.unique(unique_indices)]

    return unique_features, unique_labels

class TextDataset(Dataset):
    """自定义数据集类"""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def data_iterator(features, labels, batch_size=32, shuffle=True):
    """
    创建数据迭代器

    参数:
        features: 特征张量 (torch.Tensor)
        labels: 标签张量 (torch.Tensor)
        batch_size: 批量大小 (默认32)
        shuffle: 是否随机打乱数据 (默认True)

    返回:
        DataLoader迭代器
    """
    # 确保特征和标签长度一致
    assert len(features) == len(labels), "特征和标签数量不匹配"

    # 创建数据集
    dataset = TextDataset(features, labels)

    # 创建数据加载器
    data_iter = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False  # 不丢弃最后不足batch_size的批次
    )

    return data_iter

def shuffl_split_data(df, test_size = 0.2):
    """加载并预处理数据"""

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,       # 测试集比例
        stratify=df['dom']
        # random_state=42      # 随机种子
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def based_on_keywords(data_df):
    '''
              text                  dom     dom_test
    番茄炖鸡胸与藜麦需要用什么材料      0         1
    天猫精灵讲讲讲个小番茄的故事        0         1
    番茄煎鸡胸和什么主食搭配            0         1
    瘦肉粥与清炒时蔬需要剥皮吗          0         1
    瘦肉粥与清炒时蔬对减脂有帮助吗       0         1
    我要唱吕可可的来一份失恋            0        1
    '''
    # 定义点餐关键词列表
    pos_keywords = ['来碗', '来份', '要碗', '来一份', '要份', "要个", "养胃", "热乎", "不费牙", "顺口的", "少盐的", "油腻的", "不腻的", "易嚼的", "少盐少糖的", "太咸的", "不要辣", "不要油腻"]
    
    # 定义非点餐关键词列表
    neg_keywords = ['播放', '等于', '声音', '音量', '乘以', "多少", "音乐", "除以", "歌曲", "打开", "唱", "脑筋急转弯"]
    
    # 创建临时列
    pos_match = data_df['text'].str.contains('|'.join(pos_keywords))
    neg_match = data_df['text'].str.contains('|'.join(neg_keywords))
    
    # 初始化全部为 -1
    data_df['dom_key'] = -1
    
    # 仅包含 pos_keywords 的记为 1
    data_df.loc[pos_match & ~neg_match, 'dom_key'] = 1
    
    # 仅包含 neg_keywords 的记为 0
    data_df.loc[~pos_match & neg_match, 'dom_key'] = 0
    
    # 同时包含 pos 和 neg 的保持 -1（无需额外操作，因为初始值已经是-1）

    return data_df

# 示例用法
if __name__ == "__main__":
    
    data = pd.read_csv("/mnt/disk/wjh23/EaseDineDatasets/智慧养老_label/train.txt",sep="\t")
    data = process_data(data)
    train_df, test_df = train_test_split(
            data,
            test_size=0.15,       # 测试集比例
            stratify=data['dom'],
            random_state=42      # 随机种子
        )

    train_em = Embeding(train_df)
    # Word3Vec 测试
    DIM = 100
    train_em.train_word2vec(DIM)
    # word2vec_embeding = em.get_word2vec(f"./EaseDine/DOM/embeding_models/word2vec/word2vec_model_0_{DIM}.bin")
    # print(word2vec_embeding[:5])

    # LDA 测试
    num_topics = 20
    train_em.train_lda(num_topics = num_topics)
    # lda_model_path = f"./EaseDine/DOM/embeding_models/lda/lda_model_{num_topics}.model"
    # lda_dictionary_path = f"./EaseDine/DOM/embeding_models/lda/lda_dictionary_{num_topics}.gensim"
    # lda_corpus_path = f"./EaseDine/DOM/embeding_models/lda/lda_corpus_{num_topics}.mm"
    # lda_embeding = em.get_lda(5,lda_model_path,lda_dictionary_path,lda_corpus_path)
    # print(lda_embeding[:5])

    # # 按行拼接（沿 axis=1 水平拼接）
    # combined_features = np.hstack([lda_embeding[:5], word2vec_embeding[:5]])

    # print("拼接后的数组形状:", combined_features.shape)
    # print("\n拼接后的数组示例（第一行）:")
    # print(combined_features)