import speech_recognition as sr

r = sr.Recognizer() #调用识别器
test = sr.AudioFile("test1.wav") #导入语音文件
with test as source:
    audio = r.record(source)
type(audio)
c=r.recognize_sphinx(audio, language='zh-cn') #中文识别输出
print("识别结果：",c)

# 从阿里云源升级pip
# python.exe -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
#
# 从阿里云源安装 SpeechRecognition 库
# pip install SpeechRecognition -i https://mirrors.aliyun.com/pypi/simple/
#
# 从阿里云源安装 pocketsphinx 库
# pip install pocketsphinx -i https://mirrors.aliyun.com/pypi/simple/
#
# C:\Users\Administrator\PycharmProjects\pythonyysb\venv\Lib\site-packages\speech_recognition\pocketsphinx-data\放置英文语言模型文件，英文发音字典（en-US）和中文语言模型文件，中文发音字典（zh-CN）
#
# C:\Users\Administrator\PycharmProjects\pythonyysb\venv\Lib\site-packages\speech_recognition\pocketsphinx-data\en-US（英文识别文件）
#
# C:\Users\Administrator\PycharmProjects\pythonyysb\venv\Lib\site-packages\speech_recognition\pocketsphinx-data\zh-CN（中文识别文件）