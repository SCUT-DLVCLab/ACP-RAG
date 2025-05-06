import json
import torch
import os
from tqdm import tqdm

# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda")

# 对古诗词进行去重
def remove_duplicates(A):
    # 计算两个列表的交集长度
    def get_common_length(list1, list2):
        return len(set(list1) & set(list2))  # 使用集合交集提高效率

    unique_contents = []  # 存储去重后的结果列表

    for dict_A in tqdm(A):  # 遍历所有待处理数据
        content = dict_A["Content_split"]
        should_remove = False  # 标记当前项是否应被过滤

        # 遍历当前已收录的唯一内容，进行相似度比较
        for i in range(len(unique_contents)):
            unique_content = unique_contents[i].get("Content_split")

            # 去除每个 item 的最后一个字符，进行比较
            content_1 = [item[:-1] for item in content]
            unique_content_1 = [item[:-1] for item in unique_content]

            # 计算两个内容的相似程度（交集长度）
            common_length = get_common_length(content_1, unique_content_1)

            # 如果二者交集长度达到较短一方的一半及以上，视为重复
            if common_length >= min(len(set(content_1)), len(set(unique_content_1))) / 2:
                # 若当前内容更长，则替换原内容
                if len(content) > len(unique_content):
                    unique_contents[i] = dict_A
                    should_remove = True
                else:
                    should_remove = True
                break  # 已判断为重复，无需再继续比对

        # 如果不重复，则加入唯一内容列表
        if not should_remove:
            unique_contents.append(dict_A)

        # 每处理 100 项，保存一次中间结果
        if dict_A["Id"] % 100 == 0:
            with open("path/to/output_file.json", "w", encoding="utf-8") as f:
                json.dump(unique_contents, f, ensure_ascii=False, indent=2)

    return unique_contents

# 读取输入文件（路径需补全）
with open("path/to/input_file.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 执行去重处理
unique_list = remove_duplicates(data)

# 将去重后的结果写入输出文件
with open("path/to/output_file.json", "w", encoding="utf-8") as f:
    json.dump(unique_list, f, ensure_ascii=False, indent=2)
