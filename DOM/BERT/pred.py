from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time

t0 = time.time()
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('E:/githubworkspace/EaseDine/DOM/BERT/final_model')
# 加载模型
model = AutoModelForSequenceClassification.from_pretrained('E:/githubworkspace/EaseDine/DOM/BERT/final_model')
print(f"加载模型耗时：{time.time() - t0}s.")

# 示例输入
# text = "小明，明天我需要上班，记得把家里的衣服收一下。"
text = ["小明，明天我需要上班，记得把家里的衣服收一下。",
        "节日大优惠，限购100件，快来抢购，拨打电话188XXX",
        "哈哈哈哈，无语了",
        "聚焦企业‘造血能力’，严选‘现金奶牛’企业。点击了解："]
t1 = time.time()
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
# 进行预测
outputs = model(**inputs)
logits = outputs.logits
predictions = logits.argmax(dim=-1)
print(f"预测耗时：{time.time() - t1}s.")
print("预测输出：",predictions.cpu(),"[1:正常邮件；0：垃圾邮件]")

'''
加载模型耗时：1.0578579902648926s.
预测耗时：0.2296619415283203s.
预测输出： tensor([1, 0, 1, 0]) [1:正常邮件；0：垃圾邮件]
'''
