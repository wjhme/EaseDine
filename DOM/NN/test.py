# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
print(torch.cuda.is_available())
def count_records(file_path):
    """
    统计 txt 文件中的记录数（假设每条记录占一行）

    :param file_path: txt 文件的路径
    :return: 记录数
    """
    try:
        with open(file_path, 'r', encoding='gbk') as file:
            record_count = sum(1 for line in file)  # 逐行统计
        return record_count
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return 0


# # 示例调用
# ham_file = "rawData/ham.txt"
# spam_file = "rawData/spam.txt"
# word2vec_file = "rawData/word2vec.txt"
# # ham_records = count_records(ham_file)
# # spam_records = count_records(spam_file)
# # word2vec_records = count_records(word2vec_file)
# # print(f"ham_records:{ham_records}")
# # print(f"spam_records:{spam_records}")
# # print(f"word2vec_records:{word2vec_records}")
# with open(word2vec_file, 'r', encoding='gbk') as file:
#     for line in file:
#         print(line)
#         break