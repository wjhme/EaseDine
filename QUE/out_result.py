import pandas as pd
from pathlib import Path
import re
import os

# 初始化路径
cur_dir = Path(__file__).parent
root_dir = cur_dir.parent.parent

# 读取分类后的数据
dom_df = pd.read_csv(f"{root_dir}/EaseDineDatasets/A_audio_dom_model1.txt",sep="\t")
dom_df.loc[dom_df['dom']==-1,'dom'] = 0

# 读取推荐菜品后的数据
que_df = pd.read_csv(f"{root_dir}/EaseDineDatasets/df_query_food_feature.txt",sep=",",header=None,names=['uuid', 'text', 'dom', 'food'])
que_df['dom'] = que_df['dom'].astype(int)

# print(dom_df.head(10))
# print(que_df.head(10))

# 使用 merge 合并数据，保留 dom_df 的所有行（left join）
result_df = dom_df.merge(que_df[['uuid', 'food']], on='uuid', how='left')
# 删除字母和空格
result_df['text'] = result_df['text'].str.replace(r'[a-zA-Z\s]', '', regex=True)

# 读取A.txt
# df = pd.read_csv("/mnt/disk/wjh23/EaseDineDatasets/Result.txt",sep="\t",header=None,names=['uuid', 'text', 'dom', 'food'])
A_df = pd.read_csv("/mnt/disk/wjh23/EaseDineDatasets/智慧养老_label/A.txt",sep="\t")

result_df = A_df.merge(result_df, on='uuid', how='left')[['uuid', 'text_y', 'dom_y', 'food']]

# print(result_df)
result_df.to_csv(f"{root_dir}/EaseDineDatasets/Results.txt", sep='\t', header=None, index=False, na_rep='')