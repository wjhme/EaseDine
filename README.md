# EaseDine
智慧点餐系统，实现精准识别用户语音内容，并判断用户是否有点餐意图，针对点餐指令合理化推荐外卖菜品。

## 功能特性

- **核心功能1-ASR**：高精度语音识别

模型文件：[智慧养老语音识别模型 · 模型库](https://www.modelscope.cn/models/wjh6002/speech_recognition_moel)

方法：数据增强+FunASR开源语音识别模型微调
- **核心功能2-DOM**：用户意图识别

规则判别+集成模型分类（基分类器:naive_bayes,logistic_regression,svm,random_forest）
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