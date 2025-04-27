import os
import time
import dashscope

user_input = "我要唱张杰的看了看"
messages = [
    {'role': 'user', 'content': 
f"""
  "任务要求": {
    "操作规范": {
      "严格单字替换": "仅允许修改经双重验证的错别字（音近、形近）",
      "结构保全": {
        "字符长度锁定": "绝对禁止增减字符",
        "格式保留": "保留所有标点、空格及数字格式"
      },
      "专业术语处理": {
        "知识库优先": "调用内置专业词库验证（含音乐、餐饮、地理等领域）",
        "多版本兼容": "对存在多版本译名的专有名词维持原状"
      },
      "方言保护机制": "对明显方言特征表述启动保护性跳过"
    },
    "质量保障": {
      "置信度阈值": "仅当纠错置信度>90%时执行修改",
      "上下文关联分析": "启用N-gram语言模型验证语义合理性",
      "版本追踪": "保留原始文本的完整修订记录"
    }
  },
  "示例演示": [
    {
      "输入": "播放周杰伦的青花次",
      "分析过程": [
        "『次』字在歌曲库中无匹配项",
        "音近匹配『瓷』置信度98%",
        "验证『青花瓷』为正式曲目"
      ],
      "输出": {
        "Original_text": "播放周杰伦的青花次",
        "Text_corrected": "播放周杰伦的青花瓷"
      }
    },
    {
      "输入": "重庆的白宫馆监欲在哪里",
      "分析过程": [
        "『宫』在地名词库中应为『公』置信度95%",
        "『欲』字不符合监狱场景特征",
        "『狱』字匹配度99%且符合字符定位"
      ],
      "输出": {
        "Original_text": "重庆的白宫馆监欲在哪里",
        "Text_corrected": "重庆的白公馆监狱在哪里"
      }
    }
  ],
  "输入文本": "{user_input}"
"""}
]

t0 = time.time()
response = dashscope.Generation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model="deepseek-v3",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
    messages=messages,
    # result_format参数不可以设置为"text"。
    result_format='message'
)
print(f"{time.time() - t0:.4f}")

# print("=" * 20 + "思考过程" + "=" * 20)
# print(response.output.choices[0].message.reasoning_content)
# print(f"{time.time() - t0:.4f}")

print("=" * 20 + "最终答案" + "=" * 20)
print(response.output.choices[0].message.content)

print(f"{time.time() - t0:.4f}")