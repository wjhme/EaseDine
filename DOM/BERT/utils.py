import torch

class SpamDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: val[idx]
            for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 文本编码
def encode_texts(tokenizer, texts, max_length):
    encodings = tokenizer(
        texts.tolist(), 
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 新增：检查input_ids是否越界
    vocab_size = tokenizer.vocab_size
    input_ids = encodings['input_ids']
    if (input_ids >= vocab_size).any():
        invalid_indices = torch.where(input_ids >= vocab_size)
        print(f"错误：发现 {len(invalid_indices[0])} 个超出词表的索引（词表大小={vocab_size}）")
        print("示例无效文本:", texts[invalid_indices[0][0].item()])
        raise ValueError("Tokenization生成无效的input_ids")
        
    return encodings

# 预测
def predict_spam(model, tokenizer, text, max_length=128):
    # 切换到评估模式
    model.eval()

    # 文本编码与设备转移
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(model.device)  # 关键：确保输入与模型同设备

    # 推理过程（禁用梯度计算）
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取预测概率
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0].cpu().numpy()  # 返回CPU端的numpy数组