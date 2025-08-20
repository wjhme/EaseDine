# EaseDine
本项目研发面向老年用户的低门槛外卖语音交互系统，针对方言口音（如粤语、川渝方言）和模糊表达（如“少放辣”“炖烂点”）等痛点，通过语音识别和语义理解技术，实现语音点餐的精准交互，降低操作门槛，帮助老年人轻松使用外卖服务。

![EaseDine方案](E:\githubworkspace\EaseDine\EaseDine方案.png)

## 功能特性

- **核心功能1-ASR**：高精度语音识别

模型文件：[智慧养老语音识别模型 · 模型库](https://www.modelscope.cn/models/wjh6002/speech_recognition_moel)

方法：数据增强+FunASR开源语音识别模型微调
- **核心功能2-DOM**：用户意图识别

规则判别+集成模型分类
- **核心功能3-QUE**：饮食智能推荐

LLM提取指令中菜品特征+基于Faiss进行相似菜品召回+LLM推荐

## 快速开始

### 前置条件

- Python 3.10
- pip 20.0+

### 安装指南

```bash
cd project

# 创建虚拟环境（推荐） Linux
conda activate -n easedine python=3.10
conda activate easedine

# 安装依赖
pip install -r requirements.txt

# 运行
python run.py
```