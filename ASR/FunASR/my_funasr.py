import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Union

import torch
import torchaudio
import pandas as pd
from pydub import AudioSegment
from funasr import AutoModel
from .utils import save_results_to_txt


class ASR:
    """语音识别系统封装类
    
    Attributes:
        model_name (str): FunASR模型名称
        model_revision (str): 模型版本号
        model_cache_dir (str): 模型缓存目录
        device (str): 计算设备(cpu/cuda)
        model: 初始化的ASR模型实例
    """

    # model_name: finetuner_mabin | finetuner_enhanced_cer_over_0 | finetuner_mabin_enhanced_cer_over_0
    
    def __init__(
        self,
        model_name: str = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        model_revision: str = "v2.0.4",
        model_cache_dir: str = "",
        device: Optional[str] = None,
    ):
        """初始化语音识别模型
        
        Args:
            model_name: FunASR模型名称，默认为paraformer-large版本
            model_revision: 模型版本号
            model_cache_dir: 自定义模型缓存路径
            device: 强制指定计算设备(None时自动检测)
        """
        # 配置环境变量
        os.environ["MODELSCOPE_CACHE"] = model_cache_dir
        os.environ["MODELSCOPE_HUB_CACHE"] = model_cache_dir
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        
        # 设备配置
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True  # 启用CUDA加速
            
        # 模型初始化
        t0 = time.time()
        self.model = AutoModel(
            model=model_name,
            model_revision=model_revision,
            cache_dir=model_cache_dir,
            # vad_model="fsmn-vad", 
            # vad_kwargs={"max_single_segment_time": 60000},
            disable_update=True,
            device=self.device,
            punc_config={"enable": False}  # 禁用标点预测
        )
        # self.model_vad = AutoModel(
        #     model=model_name,
        #     model_revision=model_revision,
        #     cache_dir=model_cache_dir,
        #     vad_model="fsmn-vad", 
        #     vad_kwargs={"max_single_segment_time": 60000},
        #     disable_update=True,
        #     device=self.device,
        #     punc_config={"enable": False}  # 禁用标点预测
        # )
        print(f"模型加载耗时: {time.time() - t0:.2f}s | 设备: {self.device.upper()}")

    def load_audio(
        self,
        audio_path: Union[str, Path],
        target_sr: int = 16000
    ) -> torch.Tensor:
        """加载并预处理音频
        
        Args:
            audio_path: 音频文件路径
            target_sr: 目标采样率(默认16kHz)
            
        Returns:
            waveform: 预处理后的音频张量(shape: [1, T])
            
        Raises:
            FileNotFoundError: 音频文件不存在
            RuntimeError: 音频处理失败
        """
        audio_file = Path(audio_path).expanduser().resolve()
        if not audio_file.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_file}")

        try:
            waveform, sample_rate = torchaudio.load(str(audio_file))
            
            # 采样率转换
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=target_sr
                )
                waveform = resampler(waveform)
            
            # 单声道处理
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
            return waveform
        
        except Exception as e:
            raise RuntimeError(f"音频处理失败: {str(e)}")

    def transcribe(
        self,
        audio_input: Union[str, Path, torch.Tensor],
        sample_rate: int = 16000,
        beam_size: int = 1
    ) -> str:
        """执行语音识别
        
        Args:
            audio_input: 输入音频（支持文件路径或已加载的张量）
            sample_rate: 音频采样率（当输入为张量时需指定）
            beam_size: 束搜索宽度
            
        Returns:
            识别文本结果
        """
        # 自动判断输入类型
        if isinstance(audio_input, (str, Path)):
            waveform = self.load_audio(audio_input)
            sr = 16000  # load_audio确保输出16kHz
        else:
            waveform = audio_input
            sr = sample_rate
            
        # 转换为numpy数组
        waveform_np = waveform.squeeze().numpy()
        
        # # 根据音频长度选择模型执行识别
        # audio = AudioSegment.from_file(audio_input)
        # duration_ms = len(audio)  # 音频总时长（毫秒）
        # print(f"duration_ms:{duration_ms}")
        # if duration_ms <= 10.5 * 1000: # 时长小于等于10s使用原始模型
        result = self.model.generate(
            input=waveform_np,
            sample_rate=sr,
            beam_size=beam_size
        )
            # print(result[0]['text'])
        # else:# 时长大于10s使用原始模型和含vad模型分别识别，若vad识别文本与原始模型识别文本长度差大于1，使用vad文本
        #     result_raw = self.model.generate(
        #         input=waveform_np,
        #         sample_rate=sr,
        #         beam_size=beam_size
        #     )
        #     result_vad = self.model_vad.generate(
        #         input=waveform_np,
        #         sample_rate=sr,
        #         beam_size=beam_size
        #     )
        #     result = result_vad if (len(result_vad[0]["text"]) - len(result_raw[0]["text"])) > 2 else result_raw
        #     print(result[0]['text'])
        
        return result[0]["text"] if result else "天猫精灵"

    def process_file(
        self,
        audio_path: Union[str, Path]
    ) -> Dict[str, Union[str, float]]:
        """处理单个音频文件
        
        Returns:
            包含识别结果、耗时等信息的字典
        """
        result = {"file": str(audio_path), "status": "failed", "text": "", "duration": 0.0}
        
        try:
            t0 = time.time()
            text = self.transcribe(audio_path)
            result.update({
                "status": "success",
                "text": text,
                "duration": time.time() - t0
            })
        except Exception as e:
            result["text"] = str(e)
            
        return result

    def batch_process(
        self,
        input_path: Union[str, Path, List[Union[str, Path]]],
        output_file: str = "results.txt"
    ) -> pd.DataFrame:
        """批量处理音频文件
        
        Args:
            input_path: 输入路径（支持目录/文件/文件列表）
            output_file: 结果保存路径
            
        Returns:
            包含所有识别结果的DataFrame
        """
        # 解析输入文件列表
        if isinstance(input_path, (str, Path)):
            path = Path(input_path)
            if path.is_dir(): # 音频目录
                audio_files = sorted(path.glob("*.wav")) + sorted(path.glob("*.mp3")) + sorted(path.glob("*.flac"))
            else: # 音频路径
                audio_files = [path]
        else: # 音频路径列表
            audio_files = [Path(p) for p in input_path]
            
        print(f"开始处理 {len(audio_files)} 个文件...")
        
        # 执行识别
        results = []
        total_start = time.time()
        for idx, audio_file in enumerate(audio_files, 1):
            print(f"处理进度: {idx}/{len(audio_files)} | 当前文件: {audio_file.name}")
            results.append(self.process_file(audio_file))
            
        # 转换为DataFrame
        df = pd.DataFrame(results)
        df.rename(columns={
            "file": "uuid",
            "duration": "time",
            "text": "text",
            "status": "status"
        }, inplace=True)
        
        # 保存结果
        save_results_to_txt(df, output_file)
        
        # 打印统计信息
        total_time = time.time() - total_start
        print(f"\n处理完成！总耗时: {total_time:.2f}s")
        print(f"平均耗时: {df['time'].mean():.2f}s/文件")
        print(f"结果保存至: {output_file}")
        
        return df


if __name__ == "__main__":
    # nohup python FunASR.py >> log/fangyan_finetuner.log 2>&1 &

    # # 启动前预加载模型（可选）
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    
    asr = ASR()
    # # 批处理
    # input_path = "/mnt/disk/wjh23/EaseDineDatasets/train_audio/train_audio_batch_1"  # 替换为你的音频文件/目录路径
    # # input_path = "/mnt/disk/wjh23/EaseDineDatasets/无法识别语音"
    # asr.batch_process(
    #     input_path=input_path,
    #     output_file="/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_all_batch_results/uuid_hypothesis/batch_1_finetuner_2345_6.txt"
    # )

    fangyan = "/mnt/disk/wjh23/filtered_results.txt"
    fangyan_df = pd.read_csv(fangyan, sep="\t")
    uuid_ls = fangyan_df['uuid'].tolist()

    import json
    uuid_dict_path = "/mnt/disk/wjh23/EaseDineDatasets/智慧养老_label/audio_paths.json"
    with open(uuid_dict_path, 'r', encoding='utf-8') as f:
        uuid_dict = json.load(f)
    
    audio_files = []
    for audio in uuid_ls:
        audio_files.append(uuid_dict[audio])

    asr.batch_process(
        audio_files, 
        output_file="/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_all_batch_results/uuid_hypothesis/fangyan_finetuner_2345_6.txt"
    )

    # # 单文件处理
    # input_path = "/mnt/disk/wjh23/EaseDineDatasets/train_audio_batch_1/0a8a651b-c341-40ca-bd79-194c4a39bfb6.wav"
    # process_single_file(input_path)
