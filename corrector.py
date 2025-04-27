# import time

# import torch
# from transformers import BertTokenizerFast, BertForMaskedLM


# device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# tokenizer = BertTokenizerFast.from_pretrained("shibing624/macbert4csc-base-chinese")
# model = BertForMaskedLM.from_pretrained("shibing624/macbert4csc-base-chinese")
# model.to(device)
# print(device)

# t0 = time.time()

# texts = ["我想听任鹏鹏的四处旁煌", "我想听自动挡吉他手的爱海涛涛","我想听杨一然的遗悍接在风中"]
# with torch.no_grad():
#     outputs = model(**tokenizer(texts, padding=True, return_tensors='pt').to(device))

# result = []
# for ids, text in zip(outputs.logits, texts):
#     _text = tokenizer.decode(torch.argmax(ids, dim=-1), skip_special_tokens=True).replace(' ', '')
#     corrected_text = _text[:len(text)]
#     print(text, ' => ', corrected_text)
#     result.append(corrected_text)
# # print(result)

# print(f"{time.time() - t0:.2f}")

import os
from pycorrector.gpt.gpt_corrector import GptCorrector
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    error_sentences = [
        '我想听杨一然的遗悍接在风中',
        '我想听自动挡吉他手的爱海涛涛',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '我想听任鹏鹏的四处旁煌',
        '我想要紫菜淡花汤',
    ]
    m = GptCorrector("shibing624/chinese-text-correction-7b", pad_token="<pad>" )

    batch_res = m.correct_batch(error_sentences)
    for i in batch_res:
        print(i)
        print()

# def clean_output(text: str) -> str:
#     # 移除所有对话标记
#     markers = ["<|im_start|>", "<|im_end|>", "system", "user", "assistant"]
#     for marker in markers:
#         text = text.replace(marker, "")
#     return text.strip()

# # pip install transformers
# from transformers import AutoModelForCausalLM, AutoTokenizer
# checkpoint = "shibing624/chinese-text-correction-7b"

# device = "cuda" # for GPU usage or "cpu" for CPU usage
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# input_content = "文本纠错：\n我想听杨一然的遗悍接在风中"

# messages = [{"role": "user", "content": input_content}]
# input_text=tokenizer.apply_chat_template(messages, tokenize=False)

# print("input_text:",input_text)

# # inputs = tokenizer.encode(input_text, return_tensors="pt",padding=True,truncation=True).to(device)
# inputs = tokenizer(
#     input_text, 
#     return_tensors="pt",  # 返回 PyTorch 张量
#     padding=True,          # 自动填充到相同长度
#     truncation=True        # 截断到模型最大长度
# ).to(device)

# input_ids = inputs.input_ids
# attention_mask = inputs.attention_mask
# outputs = model.generate(input_ids=input_ids,attention_mask=attention_mask, max_new_tokens=1024, temperature=None, do_sample=False,top_p=None,top_k=None, repetition_penalty=1.08)

# print("output:",clean_output(tokenizer.decode(outputs[0])))
