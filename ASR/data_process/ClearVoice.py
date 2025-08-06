from clearvoice import ClearVoice
import time

myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormerGAN_SE_16K'])


# # 处理单个音频文件
# input_path='/mnt/disk/wjh23/EaseDineDatasets/存在噪音的语音/00a9d51d-7a9b-4d9a-8d1a-6b45c8771c20.wav'
# output_wav = myClearVoice(input_path=input_path, online_write=False)
# # 保存增强后的音频
# output_path='/mnt/disk/wjh23/EaseDineDatasets/处理后音频/00a9d51d-7a9b-4d9a-8d1a-6b45c8771c20.wav'
# myClearVoice.write(output_wav, output_path=output_path)

t0 = time.time()
# #process wave directory
myClearVoice(input_path='/mnt/disk/wjh23/EaseDineDatasets/dataset_b', online_write=True, output_path='/mnt/disk/wjh23/EaseDineDatasets/clean_dataset_b')
print(f"{time.time() - t0}")

# #process wave list file
# myClearVoice(input_path='samples/scp/audio_samples.scp', online_write=True, output_path='samples/path_to_output_wavs_scp')