import json
from tqdm import tqdm

# 去重函数：根据 "Content_split" 字段的内容（去除每项的最后一个字符）生成哈希值以去重
def remove_duplicates(data):
    unique_contents = []         # 用于存储去重后的数据
    unique_hashes = set()        # 用于存储已出现的哈希值，便于查重

    # 初始化哈希集合（如果 unique_contents 非空，可用于增量去重）
    for j in range(len(unique_contents)):
        content = tuple(item[:-1] for item in unique_contents[j]["Content_split"])  # 去除每项的最后一个字符后转为元组
        content_hash = hash(content)  # 生成哈希值
        unique_hashes.add(content_hash)

    # 遍历输入数据，判断是否为重复项
    for dict_A in tqdm(data):
        content = tuple(item[:-1] for item in dict_A["Content_split"])  # 同样处理每项内容
        content_hash = hash(content)
        if content_hash not in unique_hashes:
            unique_contents.append(dict_A)  # 添加非重复项
            unique_hashes.add(content_hash)  # 记录该项的哈希值

    return unique_contents

# 将处理后的数据批量写入 JSON 文件
def batch_write_to_file(unique_list, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique_list, f, ensure_ascii=False, indent=2)  # 保持中文和格式化输出
    print("总数量:", len(unique_list))

# 主函数：读取输入文件，去重处理后写入输出文件
def main(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    unique_list = remove_duplicates(data)
    batch_write_to_file(unique_list, output_file)

if __name__ == "__main__":
    input_file = "path/to/input_file.json"
    output_file = "path/to/output_file.json"
    main(input_file, output_file)
