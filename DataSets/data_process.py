import pandas as pd

# 处理train.txt，去除dom列，生成标注文件用于微调FireRedASR
def generate_annotation_file(file_dir, save_dir):
    df = pd.read_csv(file_dir,sep='\t')
    df.drop('dom',axis=1,inplace=True)
    # 删除字母和空格
    df['text'] = df['text'].str.replace(r'[a-zA-Z\s，。？！]', '', regex=True)
    df.to_csv(save_dir, header=False,index=None,sep="\t")


if __name__=="__main__":
    file_dir = "/mnt/disk/wjh23/EaseDineDatasets/智慧养老_label/train.txt"
    save_dir = "/mnt/disk/wjh23/EaseDineDatasets/train_audio/transcript.txt"
    generate_annotation_file(file_dir, save_dir)