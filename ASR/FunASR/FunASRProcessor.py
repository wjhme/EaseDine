import os
import torch
import time
import pandas as pd
import torchaudio
from pathlib import Path
from typing import Union
from funasr import AutoModel


class FunASRProcessor:
    """FunASR语音识别处理器"""

    def __init__(self,
                 model_name: str = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                 model_revision: str = "v2.0.4",
                 cache_dir: str = "/mnt/disk/wjh23/models/FunASR_model",
                 device: str = None):
        """
        初始化语音识别处理器
        :param model_name: 模型名称
        :param model_revision: 模型版本
        :param cache_dir: 模型缓存目录
        :param device: 指定计算设备 (cuda/cpu)
        """
        # 硬件配置
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        # 模型配置
        self.model_name = model_name
        self.model_revision = model_revision
        self.cache_dir = Path(cache_dir).expanduser()

        # 初始化模型
        self._initialize_model()

    def _initialize_model(self):
        """加载预训练模型"""
        os.environ['MODELSCOPE_CACHE'] = str(self.cache_dir)

        t_start = time.time()
        self.model = AutoModel(
            model=self.model_name,
            model_revision=self.model_revision,
            cache_dir=str(self.cache_dir),
            disable_update=True,
            device=self.device,
            punc_config={"enable": False},
        )
        print(f"模型加载完成，耗时 {time.time() - t_start:.2f}s | 设备: {self.device}")

    @staticmethod
    def load_audio(audio_path: Union[str, Path], target_sr: int = 16000) -> torch.Tensor:
        """
        加载并预处理音频
        :param audio_path: 音频文件路径
        :param target_sr: 目标采样率
        :return: 音频张量 (1, samples)
        """
        audio_file = Path(audio_path).resolve()
        if not audio_file.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_file}")

        waveform, sample_rate = torchaudio.load(audio_file)

        # 采样率转换
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sr
            )
            waveform = resampler(waveform)

        # 确保单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform.squeeze()

    def transcribe(self, audio_path: str) -> dict:
        """
        执行语音识别
        :param audio_path: 音频文件路径
        :return: 包含结果的字典
        """
        try:
            # 加载音频
            waveform = self.load_audio(audio_path)

            # 执行识别
            t_start = time.time()
            result = self.model.generate(
                input=waveform.numpy(),
                sample_rate=16000  # 注意根据实际采样率调整
            )

            return {
                "file": audio_path,
                "text": result[0]["text"] if result else "",
                "duration": time.time() - t_start,
                "status": "success"
            }
        except Exception as e:
            return {
                "file": audio_path,
                "text": str(e),
                "duration": 0,
                "status": "failed"
            }

    def batch_process(self, input_path: Union[str, Path], output_file: str = "results.tsv") -> None:
        """
        批量处理音频文件
        :param input_path: 输入路径（文件/目录）
        :param output_file: 输出文件路径
        """
        # 获取文件列表
        input_path = Path(input_path)
        if input_path.is_dir():
            audio_files = [f for f in input_path.glob("*") if f.suffix.lower() in ('.wav', '.mp3', '.flac')]
        elif input_path.is_file():
            audio_files = [input_path]
        else:
            raise ValueError("无效的输入路径")

        # 执行批量处理
        results = []
        total = len(audio_files)
        print(f"开始批量处理，共 {total} 个文件...")

        for idx, audio_file in enumerate(audio_files, 1):
            print(f"\n处理进度: {idx}/{total}")
            result = self.transcribe(str(audio_file))
            results.append(result)
            print(f"状态: {result['status']} | 耗时: {result['duration']:.2f}s")

        # 保存结果
        self._save_results(results, output_file)
        print(f"\n处理完成！结果已保存至: {output_file}")

    def _save_results(self, results: list, output_file: str) -> None:
        """处理结果保存逻辑"""
        df = pd.DataFrame(results)

        # 列重命名与处理
        df.rename(columns={'file': 'uuid', 'duration': 'time'}, inplace=True)
        df['uuid'] = df['uuid'].str.extract(r'([a-f0-9-]{36})')  # 提取UUID

        # 输出格式处理
        df[['uuid', 'text', 'time']].to_csv(
            output_file,
            sep="\t",
            index=False,
            float_format="%.2f"
        )