from fireredasr.models.fireredasr import FireRedAsr
from pathlib import Path
import time

CUR_DIR = Path(__file__).parent 
ROOT_DIR = CUR_DIR.parent.parent

# 语音文件代号
batch_uttid = ["test2"]
# 语音文件路径
batch_wav_path = [f"{ROOT_DIR}/ASR/audios/test2.wav"]

# FireRedASR-AED 
model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L") #需要下载
# model = FireRedAsr.from_pretrained("aed", f"{ROOT_DIR}/models/FireRedASR_pretrained_model/FireRedASR-AED-L")

t0 = time.time()
results, elapsed = model.transcribe(
    batch_uttid,
    batch_wav_path,
    {
        "use_gpu": 1,
        "beam_size": 3,
        "nbest": 1,
        "decode_max_len": 0,
        "softmax_smoothing": 1.0,
        "aed_length_penalty": 0.0,
        "eos_penalty": 1.0
    }
)
print("\n",results)
print(f"用时：{elapsed:.2f} s")

'''
load model time:10.25 s

 [{'uttid': 'test2', 'text': '甚至出现交易几乎停滞的情况', 'wav': '/mnt/disk/wjh23/EaseDine/ASR/audios/test2.wav', 'rtf': '0.2054'}]
用时：0.86 s
'''