from gensim.models.word2vec import Word2Vec
from utils import NeuralNetwork, loadfile, create_dictionaries
from sklearn.model_selection import train_test_split


# 导入word2vec模型
def word2vec_train(combined):
    model = Word2Vec.load('lstm_data/model/Word2vec_model.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined

# 获取训练集、验证集
def get_data(index_dict,word_vectors,combined,y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))#索引为0的词语，词向量全为0
    for word, index in index_dict.items():#从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_val, y_train, y_val = train_test_split(combined, y, test_size=0.2)
    print(x_train.shape,y_train.shape)
    return n_symbols,embedding_weights,x_train,y_train,x_val,y_val

#训练模型，并保存
def train():
    print('Loading Data...')
    combined,y=loadfile()
    print(len(combined),len(y))
    print('Tokenising...')
    combined = tokenizer(combined)
    print('Training a Word2vec model...')
    index_dict, word_vectors,combined=word2vec_train(combined)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols,embedding_weights,x_train,y_train,x_val,y_val=get_data(index_dict, word_vectors,combined,y)
    print(x_train.shape,y_train.shape)
    network = NeuralNetwork(X_train = x_train,Y_train = y_train,X_val = x_val, Y_val = y_val,vocab_dim = vocab_dim,n_symbols = n_symbols,num_classes = 2)
    network.start_trainer(x_train,y_train,x_val,y_val)

if __name__=='__main__':
    train()
