import faiss
import pickle
from tqdm import tqdm

# 构建所有分片文件的路径列表: d_vector_guwen_112w_ver1_part_1 到 part_63
file_paths = []
for i in range(1, 64):
    string = "path/to/id_vector_112w/id_vector_guwen_112w_ver1_part_{}".format(i)
    file_paths.append(string)

# assert len(file_paths) == 63

# 读取所有索引文件并添加到列表中
indexes = []
for file_path in tqdm(file_paths):
    with open(file_path, "rb") as f:
        index = pickle.load(f)
        indexes.append(index)

# 获取第一个索引的向量维度（确保所有索引一致）
index_dim = indexes[0].d

# 创建一个空的IndexFlatL2索引，用于合并所有向量
merged_index = faiss.IndexFlatL2(index_dim)

# 遍历所有索引，将其中的向量提取并添加到合并索引中
for idx in tqdm(indexes):
    vectors = idx.reconstruct_n(0, idx.ntotal)  # 提取所有向量（从第0个开始，共ntotal个）
    merged_index.add(vectors)  # 添加向量到合并索引中

# 将合并后的索引保存为FAISS格式的索引文件
output_path = "path/to/id_vector_guwen_112w_all_ver1.index"
faiss.write_index(merged_index, output_path)
