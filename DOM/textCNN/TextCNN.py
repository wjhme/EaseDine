import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 kernel_sizes=[3, 4, 5], num_filters=100,
                 dropout=0.6, weight_matrix = None, pretrained_embeddings=None):
        """
        TextCNN 文本分类模型

        参数:
            vocab_size: 词汇表大小
            embed_dim: 词向量维度
            num_classes: 分类类别数
            kernel_sizes: 卷积核尺寸列表 (默认[3,4,5])
            num_filters: 每种卷积核的数量 (默认100)
            dropout: dropout概率 (默认0.5)
            pretrained_embeddings: 预训练词向量 (可选)
        """
        super(TextCNN, self).__init__()

        # 1. 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(weight_matrix, freeze=False)

        # 2. 多尺度卷积层
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, num_filters, (k, embed_dim)),
                nn.BatchNorm2d(num_filters),  # 添加BN层
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for k in kernel_sizes  # 多尺度
        ])

        # 3. 全连接层
        # 增加全连接层宽度
        self.fc = nn.Sequential(
            nn.Linear(num_filters * len(kernel_sizes), 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # 4. Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        输入:
            x: 文本序列 (batch_size, seq_len)
        输出:
            logits: 预测结果 (batch_size, num_classes)
        """
        # 嵌入层 (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        x = self.embedding(x)

        x = F.normalize(x, p=2, dim=-1)  # L2归一化

        # 增加通道维度 (batch_size, 1, seq_len, embed_dim)
        x = x.unsqueeze(1)

        # 多尺度卷积和池化
        conv_outputs = []
        for conv in self.convs:
            # 卷积: (batch_size, num_filters, seq_len - kernel_size + 1, 1)
            conv_out = F.relu(conv(x))

            # 最大池化: (batch_size, num_filters, 1, 1)
            pool_out = F.max_pool2d(conv_out, (conv_out.size(2), 1))

            conv_outputs.append(pool_out.squeeze(-1).squeeze(-1))

        # 拼接所有卷积结果 (batch_size, num_filters * len(kernel_sizes))
        x = torch.cat(conv_outputs, 1)

        # Dropout和全连接
        x = self.dropout(x)
        logits = self.fc(x)

        return logits


# 示例用法
if __name__ == "__main__":
    # # 模拟参数
    # vocab_size = 5000  # 词汇表大小
    # embed_dim = 300  # 词向量维度
    # num_classes = 5  # 分类类别数
    # batch_size = 32  # 批量大小
    # seq_len = 70  # 文本固定长度
    #
    # # 创建模型
    # model = TextCNN(vocab_size, embed_dim, num_classes)
    # print(model)
    #
    # # 模拟输入
    # inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    #
    # # 前向传播
    # outputs = model(inputs)
    # print(f"输入形状: {inputs.shape}")
    # print(f"输出形状: {outputs.shape}")  # 应为(batch_size, num_classes)
    import torch

    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    print(f"GPU设备数: {torch.cuda.device_count()}")