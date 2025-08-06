1.运行 llm_recommend.py,将推荐结果保存至result中

2.运行 文本语义聚类5-02.ipynb,得到模糊描述的聚类结果(模糊特征.txt),用此结果针对性增加召回通道

3.运行 数据清洗.ipynb,得到商品-清洗.txt

注：1. llm_recommend.py用到：
    本地部署 开源模型BAAI/bge-large-zh-v1.5做embedding https://huggingface.co/BAAI/bge-large-zh-v1.5
    从阿里云平台调用大模型qwen2.5-32b-instruct https://bailian.console.aliyun.com/
    从阿里云平台微调并调用大模型,模型名称qwen2.5-32b-instruct_weitiao515  #微调数据集通过对训练集进行手工标注构造#

    2.运行llm_recommend.py时，粘贴api_key：sk-606b64b8196d4f119a7fb1789c86ac7c

