import os
from pathlib import Path
from modelscope import snapshot_download

CUR_DIR = Path(__file__).parent 
ROOT_DIR = CUR_DIR.parent.parent

# 模型缓存路径
custom_cache_path = "/mnt/disk/wjh23/models/FunASR_SenseVoiveSmall_model"
os.environ['MODELSCOPE_CACHE'] = custom_cache_path

model_dir = snapshot_download('damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch', revision='v2.0.4')