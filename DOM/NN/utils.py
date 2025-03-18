from gensim.corpora import Dictionary
from keras.preprocessing import sequence
import numpy as np

class NeuralNetwork(object):
    def __init__(self, X_train, Y_train, X_val, Y_val, vocab_dim, n_symbols, num_classes=2):
        paddle.init(use_gpu=with_gpu, trainer_count=1)

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.vocab_dim = vocab_dim
        self.n_symbols = n_symbols

        self.num_classes = num_classes

    # 定义网络模型
    def get_network(self):
        # 分类模型
        x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(self.vocab_dim))
        y = paddle.layer.data(name='y', type=paddle.data_type.integer_value(self.num_classes))
        fc1 = paddle.layer.fc(input=x, size=1280, act=paddle.activation.Linear())
        fc2 = paddle.layer.fc(input=fc1, size=640, act=paddle.activation.Relu())
        prob = paddle.layer.fc(input=fc2, size=self.num_classes, act=paddle.activation.Softmax())
        predict = paddle.layer.mse_cost(input=prob, label=y)


        return predict


    # 定义训练器
    def get_trainer(self):
        cost = self.get_network()

        # 获取参数
        parameters = paddle.parameters.create(cost)

        # 定义优化方法
        optimizer0 = paddle.optimizer.Momentum(
            momentum=0.9,
            regularization=paddle.optimizer.L2Regularization(rate=0.0002 * 128),
            learning_rate=0.01 / 128.0,
            learning_rate_decay_a=0.01,
            learning_rate_decay_b=50000 * 100)

        optimizer1 = paddle.optimizer.Momentum(
            momentum=0.9,
            regularization=paddle.optimizer.L2Regularization(rate=0.0002 * 128),
            learning_rate=0.001,
            learning_rate_schedule="pass_manual",
            learning_rate_args="1:1.0, 8:0.1, 13:0.01")

        optimizer = paddle.optimizer.Adam(
            learning_rate=2e-3,
            regularization=paddle.optimizer.L2Regularization(rate=8e-4),
            model_average=paddle.optimizer.ModelAverage(average_window=0.5))

        # 创建训练器
        trainer = paddle.trainer.SGD(
            cost=cost, parameters=parameters, update_equation=optimizer)
        return parameters, trainer


    # 开始训练
    def start_trainer(self, X_train, Y_train, X_val, Y_val):
        parameters, trainer = self.get_trainer()

        result_lists = []

        def event_handler(event):
            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 100 == 0:
                    print("\nPass %d, Batch %d, Cost %f, %s" % (event.pass_id, event.batch_id, event.cost, event.metrics))
            if isinstance(event, paddle.event.EndPass):
                # 保存训练好的参数
                with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
                    parameters.to_tar(f)
                # feeding = ['x','y']
                result = trainer.test(
                    reader=val_reader)
                # feeding=feeding)
                print("\nTest with Pass %d, %s" % (event.pass_id, result.metrics))
                result_lists.append((event.pass_id, result.cost,result.metrics['classification_error_evaluator']))

        # 开始训练
        train_reader = paddle.batch(paddle.reader.shuffle(
            reader_creator(X_train, Y_train), buf_size=20),
            batch_size=4)

        val_reader = paddle.batch(paddle.reader.shuffle(
            reader_creator(X_val, Y_val), buf_size=20),
            batch_size=4)

        trainer.train(reader=train_reader, num_passes=5, event_handler=event_handler)

        # 找到训练误差最小的一次结果
        best = sorted(result_lists, key=lambda list: float(list[1]))[0]
        print('Best pass is %s, testing Avgcost is %s' % (best[0], best[1]))
        print('The classification accuracy is %.2f%%' % (100 - float(best[2]) * 100))

# 加载数据
def loadfile():
    # 加载正样本
    fopen = open('rawData/ham.txt', 'r')
    pos = []
    for line in fopen:
        pos.append(line)

    # 加载负样本
    fopen = open('rawData/spam.txt', 'r')
    neg = []
    for line in fopen:
        neg.append(line)

    combined = np.concatenate((pos, neg))
    # 创建label
    y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)))
    return combined, y


# 创建paddlepaddle读取数据的reader
def reader_creator(dataset, label):
    def reader():
        for i in range(len(dataset)):
            yield dataset[i, :], int(label[i])

    return reader

#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,combined=None):
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                sentences = sentence.split(' ')
                for word in sentences:
		    try:
		        word = unicode(word, errors='ignore')
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print('No data provided...')
